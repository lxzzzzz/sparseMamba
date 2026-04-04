import torch
import torch.nn as nn
import torch.nn.functional as F


class RecoverySelectiveSSM(nn.Module):
    """
    A detector-aware selective recurrent block.
    It keeps separate motion and observability states, and lets observation
    quality control how aggressively the new evidence overwrites the memory.
    """

    def __init__(self, input_dim, hidden_dim, quality_dim, dropout=0.1):
        super().__init__()
        self.input_norm = nn.LayerNorm(input_dim)
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.local_mixer = nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1, groups=hidden_dim)
        self.quality_proj = nn.Linear(quality_dim, hidden_dim)

        self.motion_write = nn.Linear(hidden_dim * 2, hidden_dim)
        self.motion_candidate = nn.Linear(hidden_dim * 3, hidden_dim)
        self.obs_write = nn.Linear(hidden_dim * 2, hidden_dim)
        self.obs_candidate = nn.Linear(hidden_dim * 3, hidden_dim)
        self.recovery_write = nn.Linear(hidden_dim * 2, hidden_dim)
        self.recovery_candidate = nn.Linear(hidden_dim * 3, hidden_dim)
        self.out_proj = nn.Linear(hidden_dim * 3, hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, seq_input, quality_input, mask):
        seq_input = self.input_norm(seq_input)
        hidden = self.input_proj(seq_input)
        hidden = self.local_mixer(hidden.transpose(1, 2)).transpose(1, 2)
        quality_embed = self.quality_proj(quality_input)

        batch_size, seq_len, hidden_dim = hidden.shape
        motion_state = hidden.new_zeros((batch_size, hidden_dim))
        obs_state = hidden.new_zeros((batch_size, hidden_dim))
        recovery_state = hidden.new_zeros((batch_size, hidden_dim))
        outputs = []

        for step in range(seq_len):
            valid = mask[:, step:step + 1]
            x_t = hidden[:, step]
            q_t = quality_embed[:, step]

            motion_write = torch.sigmoid(self.motion_write(torch.cat([x_t, q_t], dim=-1)))
            motion_cand = torch.tanh(self.motion_candidate(torch.cat([x_t, q_t, obs_state], dim=-1)))
            obs_write = torch.sigmoid(self.obs_write(torch.cat([q_t, motion_state], dim=-1)))
            obs_cand = torch.tanh(self.obs_candidate(torch.cat([q_t, motion_state, recovery_state], dim=-1)))
            recovery_write = torch.sigmoid(self.recovery_write(torch.cat([q_t, obs_state], dim=-1)))
            recovery_cand = torch.tanh(self.recovery_candidate(torch.cat([x_t, motion_state, obs_state], dim=-1)))

            new_motion = (1.0 - motion_write) * motion_state + motion_write * motion_cand
            new_obs = (1.0 - obs_write) * obs_state + obs_write * obs_cand
            new_recovery = (1.0 - recovery_write) * recovery_state + recovery_write * recovery_cand

            motion_state = torch.where(valid > 0, new_motion, motion_state)
            obs_state = torch.where(valid > 0, new_obs, obs_state)
            recovery_state = torch.where(valid > 0, new_recovery, recovery_state)

            out_t = self.out_proj(torch.cat([motion_state, obs_state, recovery_state], dim=-1))
            outputs.append(out_t)

        outputs = torch.stack(outputs, dim=1)
        outputs = outputs + self.dropout(hidden)
        return outputs, motion_state, obs_state, recovery_state


class TrackMamba(nn.Module):
    def __init__(
        self,
        geom_dim=8,
        quality_dim=7,
        time_dim=4,
        context_dim=4,
        hidden_dim=128,
        num_blocks=2,
        dropout=0.1,
        pos_weight=6.0,
        assoc_weight=1.0,
        recovery_weight=0.5,
        survival_weight=0.3,
        motion_weight=0.4,
    ):
        super().__init__()
        self.geom_dim = int(geom_dim)
        self.quality_dim = int(quality_dim)
        self.time_dim = int(time_dim)
        self.context_dim = int(context_dim)
        self.hidden_dim = int(hidden_dim)
        self.pos_weight = float(pos_weight)
        self.loss_weights = {
            'assoc': float(assoc_weight),
            'recovery': float(recovery_weight),
            'survival': float(survival_weight),
            'motion': float(motion_weight),
        }

        track_input_dim = self.geom_dim + self.quality_dim + self.time_dim
        self.blocks = nn.ModuleList([
            RecoverySelectiveSSM(track_input_dim if idx == 0 else hidden_dim, hidden_dim, self.quality_dim, dropout=dropout)
            for idx in range(num_blocks)
        ])
        self.track_norm = nn.LayerNorm(hidden_dim)
        self.context_proj = nn.Sequential(
            nn.Linear(self.context_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.track_fuse = nn.Sequential(
            nn.Linear(hidden_dim * 4, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
        )

        self.det_encoder = nn.Sequential(
            nn.Linear(self.geom_dim + self.quality_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.next_state_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, self.geom_dim),
        )
        self.recovery_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 4),
        )
        self.survival_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim // 2, 1),
        )
        self.geom_pair_proj = nn.Linear(self.geom_dim, hidden_dim)
        self.quality_pair_proj = nn.Linear(self.quality_dim, hidden_dim)
        self.pair_head = nn.Sequential(
            nn.Linear(hidden_dim * 6, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def encode_tracks(self, batch_dict):
        geom = batch_dict['track_geom_history']
        quality = batch_dict['track_quality_history']
        time = batch_dict['track_time_history']
        mask = batch_dict['track_mask']
        context = batch_dict['track_context']

        batch_size, num_tracks, history_len, _ = geom.shape
        seq_input = torch.cat([geom, quality, time], dim=-1).view(batch_size * num_tracks, history_len, -1)
        quality_input = quality.view(batch_size * num_tracks, history_len, -1)
        seq_mask = mask.view(batch_size * num_tracks, history_len)

        x = seq_input
        motion_state = obs_state = recovery_state = None
        for block in self.blocks:
            x, motion_state, obs_state, recovery_state = block(x, quality_input, seq_mask)
        x = self.track_norm(x)

        counts = seq_mask.sum(dim=1, keepdim=True)
        pooled = (x * seq_mask.unsqueeze(-1)).sum(dim=1) / counts.clamp(min=1.0)
        context_embed = self.context_proj(context.view(batch_size * num_tracks, -1))
        track_repr = self.track_fuse(torch.cat([pooled, motion_state, obs_state, context_embed], dim=-1))
        valid_tracks = counts.squeeze(-1) > 0

        return {
            'track_repr': track_repr.view(batch_size, num_tracks, -1),
            'motion_state': motion_state.view(batch_size, num_tracks, -1),
            'obs_state': obs_state.view(batch_size, num_tracks, -1),
            'recovery_state': recovery_state.view(batch_size, num_tracks, -1),
            'valid_tracks': valid_tracks.view(batch_size, num_tracks),
        }

    def forward(self, batch_dict):
        track_outputs = self.encode_tracks(batch_dict)
        det_input = torch.cat([batch_dict['candidate_det_geom'], batch_dict['candidate_det_quality']], dim=-1)
        det_embed = self.det_encoder(det_input)

        track_repr = track_outputs['track_repr']
        next_state_pred = self.next_state_head(track_repr)
        recovery_state_logits = self.recovery_head(track_repr)
        survival_logits = self.survival_head(track_repr).squeeze(-1)

        pred_next_embed = self.geom_pair_proj(next_state_pred)
        det_geom_embed = self.geom_pair_proj(batch_dict['candidate_det_geom'])
        det_quality_embed = self.quality_pair_proj(batch_dict['candidate_det_quality'])

        num_dets = det_embed.shape[1]
        track_expand = track_repr.unsqueeze(2).expand(-1, -1, num_dets, -1)
        pred_next_expand = pred_next_embed.unsqueeze(2).expand(-1, -1, num_dets, -1)
        obs_expand = track_outputs['obs_state'].unsqueeze(2).expand(-1, -1, num_dets, -1)
        recovery_expand = track_outputs['recovery_state'].unsqueeze(2).expand(-1, -1, num_dets, -1)
        det_expand = det_embed.unsqueeze(1).expand(-1, track_repr.shape[1], -1, -1)
        det_geom_expand = det_geom_embed.unsqueeze(1).expand(-1, track_repr.shape[1], -1, -1)
        det_quality_expand = det_quality_embed.unsqueeze(1).expand(-1, track_repr.shape[1], -1, -1)

        geom_delta = torch.abs(batch_dict['candidate_det_geom'].unsqueeze(1) - next_state_pred.unsqueeze(2))
        geom_delta_embed = self.geom_pair_proj(geom_delta[..., :self.geom_dim])

        pair_feat = torch.cat([
            track_expand,
            det_expand,
            pred_next_expand - det_geom_expand,
            obs_expand * det_quality_expand,
            recovery_expand,
            geom_delta_embed,
        ], dim=-1)
        association_logits = self.pair_head(pair_feat).squeeze(-1)

        return {
            'association_logits': association_logits,
            'recovery_state_logits': recovery_state_logits,
            'survival_logits': survival_logits,
            'next_state_pred': next_state_pred,
            'valid_tracks': track_outputs['valid_tracks'],
        }

    def get_loss(self, batch_dict, output_dict):
        valid_tracks = output_dict['valid_tracks']
        assoc_mask = batch_dict['assoc_mask'] & valid_tracks.unsqueeze(-1)

        loss = output_dict['association_logits'].sum() * 0.0
        tb_dict = {}

        if assoc_mask.any():
            assoc_loss = F.binary_cross_entropy_with_logits(
                output_dict['association_logits'][assoc_mask],
                batch_dict['assoc_targets'][assoc_mask],
                pos_weight=torch.tensor(self.pos_weight, device=output_dict['association_logits'].device),
            )
            loss = loss + self.loss_weights['assoc'] * assoc_loss
            tb_dict['loss_assoc'] = float(assoc_loss.item())
        else:
            tb_dict['loss_assoc'] = 0.0

        valid_track_mask = valid_tracks
        if valid_track_mask.any():
            recovery_loss = F.cross_entropy(
                output_dict['recovery_state_logits'][valid_track_mask],
                batch_dict['recovery_state_targets'][valid_track_mask],
            )
            survival_loss = F.binary_cross_entropy_with_logits(
                output_dict['survival_logits'][valid_track_mask],
                batch_dict['survival_targets'][valid_track_mask],
            )
            loss = loss + self.loss_weights['recovery'] * recovery_loss
            loss = loss + self.loss_weights['survival'] * survival_loss
            tb_dict['loss_recovery'] = float(recovery_loss.item())
            tb_dict['loss_survival'] = float(survival_loss.item())
        else:
            tb_dict['loss_recovery'] = 0.0
            tb_dict['loss_survival'] = 0.0

        motion_mask = (batch_dict['next_geom_mask'] > 0) & valid_tracks
        if motion_mask.any():
            motion_loss = F.smooth_l1_loss(
                output_dict['next_state_pred'][motion_mask],
                batch_dict['next_geom_targets'][motion_mask],
            )
            loss = loss + self.loss_weights['motion'] * motion_loss
            tb_dict['loss_motion'] = float(motion_loss.item())
        else:
            tb_dict['loss_motion'] = 0.0

        tb_dict['loss_total'] = float(loss.item())
        return loss, tb_dict
