# Paper Framework Diagram Description

## Title
Detector-Aware Weak-Observation Recovery Mamba for Robust Online 3D Multi-Object Tracking

## Figure Goal
This figure should present the whole method as a two-stage tracking-by-detection framework for long-range sparse weak-observation 3D MOT.

The key message of the figure is:
- The detector improves single-frame visibility of weak sparse objects.
- The tracker models degradation, temporary missing observations, and recovery across time.
- Detection-side weak-observation quality is explicitly passed into the tracker.

The figure should look like a clean academic pipeline diagram, not a software architecture chart.

## Overall Layout
Use a left-to-right figure with 3 major regions:

1. Detection Stage
2. Cache / Intermediate Representation
3. Tracking Stage

Add a thin top banner showing:
- Input scenario: long-range sparse weak-observation scene
- Multi-modal input: LiDAR + Image
- Output task: robust online 3D multi-object tracking

At the bottom, add a short summary strip:
- Detection enhances visibility
- Tracking preserves identity through degradation and recovery

## Region 1: Detection Stage
Place this region on the left.

### Inputs
Show two inputs entering the detector:
- LiDAR point cloud
- RGB image

Label the scene condition near the inputs:
- sparse points
- weak observations
- projection misalignment
- score fluctuation

### Detector Backbone
Inside the detector block, show the following pipeline:

1. MeanVFE
2. VoxelNeXtFusion sparse backbone
3. Fusion at conv4
4. conv5 / conv6
5. sparse BEV aggregation
6. VoxelNeXtHead

Emphasize that the detector is still sparse:
- full sparse 3D detection backbone
- fusion inserted at stride-8 sparse feature layer

### Fusion Module Detail
Inside the detector region, add a highlighted sub-block named:
Fusion Rectifier Block

This sub-block contains:
- DeformableRectifier
- InstanceAwareGate
- fusion projection
- residual normalization

For DeformableRectifier, annotate:
- project sparse voxels to image plane
- learn local deformable sampling around projection center
- aggregate image evidence with attention

For InstanceAwareGate, annotate:
- predict foreground tendency
- suppress noisy image injection in background regions

### Detector Outputs
From the detector, draw two outputs:

1. Standard detection output:
- 3D boxes
- class scores
- labels

2. Observation-quality output:
- gate score
- valid sample ratio
- offset stability
- attention focus
- fusion gain
- aggregated reliability score

Use a grouped label:
Detector-aware weak-observation quality cues

## Region 2: Cache / Intermediate Representation
Place this region in the center as a bridge between detection and tracking.

Show this as a compact storage / memory bank block named:
Frame-wise Detector Cache

### Contents
List the cached items:
- sequence_id
- frame_idx
- pred_boxes
- pred_scores
- pred_labels
- obs_quality_vec
- reliability_scores

### Meaning
Add a small note:
- converts enhanced single-frame detection into trackable temporal observations

## Region 3: Tracking Stage
Place this region on the right.

This region should be larger than a normal tracker block because this is the paper's main innovation.

### Inputs to Tracker
Show two sources entering the tracker:

1. Historical track memory from previous frames
2. Current frame detector cache

Show the current frame candidate detections split into:
- geometry tokens
- quality tokens

Show the historical track memory split into:
- geometry history
- quality history
- time / missing-gap history
- track context

### Main Tracking Module
Use a large highlighted block named:
Detector-Aware Weak-Observation Recovery Mamba

Inside this block, show three conceptual branches:

1. Motion State Branch
- models target geometric evolution
- predicts next state

2. Observability State Branch
- models stable / degraded / missing / recovered observation conditions

3. Recovery Memory Branch
- preserves reliable history under weak observations
- supports reactivation after temporary missing detections

Add an annotation:
- selective state updates are controlled by detector-aware quality cues

### Recovery Mamba Inputs
Label the internal token types:
- geometry token
- quality token
- temporal token
- track context token

### Recovery Mamba Outputs
Show four outputs from the main tracking block:
- association logits
- recovery state logits
- survival score
- next-state prediction

## Online Association and Track Management
To the right of the Recovery Mamba, place a smaller block:
Online Association and Track Management

Show the logic as:
1. candidate filtering by class and geometry
2. Hungarian matching with learned association scores
3. recovery-aware survival management

Track states to display:
- active
- degraded
- missing
- recovered
- removed

Add a key annotation:
- dynamic keep-alive for recoverable tracks
- fewer ID switches and fewer broken trajectories

## Final Output
At the far right, show the final output:
- online 3D trajectories
- stable IDs

Also show a small sequence of boxes with the same color across frames to indicate identity consistency.

Label the final benefits:
- robust long-range tracking
- reduced ID switch
- reduced trajectory fragmentation
- better recovery after weak observation

## Core Innovation Text for Figure
If the figure includes a short central caption, use:

The detector enhances weak single-frame observations, while the proposed Recovery Mamba explicitly models degradation, missing observations, and recovery using detector-aware quality cues for robust online 3D MOT.

## Recommended Visual Emphasis
Use color grouping:
- blue for detection
- orange for weak-observation quality cues
- green for tracking / Recovery Mamba
- gray for cache and online management

Make the Recovery Mamba visually dominant.
The detector should be clearly important, but the tracker should appear as the methodological center.

## Short Caption
Figure X. Overall framework of the proposed method. The sparse multi-modal detector first enhances the visibility of long-range weak objects and exports detector-aware observation-quality cues. These cues are then injected into the proposed Recovery Mamba tracker, which jointly models motion evolution, observation degradation, temporary missing detections, and recovery for robust online 3D multi-object tracking.

## Minimal Diagram Draft
If a diagram model needs a very compact version, use this:

LiDAR + Image
-> FusionVoxelNeXt sparse detector
-> boxes + scores + detector-aware quality cues
-> frame-wise detector cache
-> Recovery Mamba tracker
-> association + recovery-aware survival management
-> online 3D trajectories with stable IDs
