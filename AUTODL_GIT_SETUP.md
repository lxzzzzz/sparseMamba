# AutoDL + Git 上手流程

这份说明按最稳的顺序写，适合第一次用 `git` 和 `AutoDL`。

## 1. 先准备远程仓库

推荐用 `Gitee` 或 `GitHub`。

你需要先在网页上完成这些动作：

1. 注册账号
2. 新建一个空仓库，例如 `sparseMamba`
3. 创建时不要勾选初始化 `README`、`.gitignore`、`LICENSE`
4. 记下仓库地址，例如：

```bash
https://gitee.com/your_name/sparseMamba.git
```

## 2. 在本地项目里初始化 Git

当前项目目录：

```bash
cd /home/lx/sparseMamba
```

依次执行：

```bash
git init
git config --global user.name "你的名字"
git config --global user.email "你的邮箱"
git status
git add .
git commit -m "init project"
```

说明：

- `.gitignore` 已经配置好，不会把常见数据、权重、缓存、日志一起提交上去。
- `git add .` 前可以先执行一次 `git status`，确认没有把不该上传的文件带进去。

## 3. 连接远程仓库并第一次推送

把下面命令里的仓库地址换成你自己的：

```bash
git remote add origin https://gitee.com/your_name/sparseMamba.git
git remote -v
git branch -M main
git push -u origin main
```

如果平台要求密码或 token，就按网页提示输入。

## 4. 在 AutoDL 上拉取代码

登录 AutoDL 后，建议把代码放在：

```bash
/root/autodl-tmp/sparseMamba
```

执行：

```bash
cd /root/autodl-tmp
git clone https://gitee.com/your_name/sparseMamba.git
cd sparseMamba
```

建议把目录整理成这样：

```text
/root/autodl-tmp/
  sparseMamba/
  data/
  cache/
  ckpt_backup/
```

其中：

- 代码：`/root/autodl-tmp/sparseMamba`
- 数据：`/root/autodl-tmp/data`
- cache：`/root/autodl-tmp/cache`

## 5. 以后本地和 AutoDL 怎么同步

### 本地改完代码

```bash
cd /home/lx/sparseMamba
git status
git add .
git commit -m "update xxx"
git push
```

### AutoDL 拉最新代码

```bash
cd /root/autodl-tmp/sparseMamba
git pull
```

建议规则：

- 本地负责改代码
- AutoDL 负责训练和评测
- 尽量不要在本地和 AutoDL 同时改同一个文件

## 6. AutoDL 上配置环境

示例：

```bash
conda create -n sparsemamba python=3.10 -y
conda activate sparsemamba
pip install -r requirements.txt
```

如果项目还依赖指定版本的 `torch`、`spconv`、`cuda`，需要按你的机器环境单独安装。

环境配好后，建议在 AutoDL 面板里保存镜像，后面换机器更省事。

## 7. 用 tmux 跑训练

不要直接在普通 SSH 窗口里长期跑训练。

新建一个会话：

```bash
tmux new -s train100
```

进入环境并开始训练：

```bash
cd /root/autodl-tmp/sparseMamba
conda activate sparsemamba
python tools/train.py --cfg_file tools/cfgs/dair_v2x_models/fusion_voxelnext_100m.yaml --extra_tag fusion_det_100m
```

后台保持运行：

```bash
Ctrl+b d
```

重新进入：

```bash
tmux attach -t train100
```

## 8. 你的训练顺序

### 100m

1. 训练检测

```bash
python tools/train.py \
  --cfg_file tools/cfgs/dair_v2x_models/fusion_voxelnext_100m.yaml \
  --extra_tag fusion_det_100m
```

2. 生成 tracking infos

```bash
python tools/create_tracking_infos.py \
  --data_path /root/autodl-tmp/data/dair_v2x_tracking \
  --save_path /root/autodl-tmp/data/dair_v2x_tracking \
  --splits train val
```

3. 生成 detector cache

```bash
python tools/generate_tracking_cache.py \
  --detector_cfg tools/cfgs/dair_v2x_models/fusion_voxelnext_100m.yaml \
  --data_cfg tools/cfgs/dataset_configs/dair_v2x_tracking_dataset_100m.yaml \
  --ckpt /path/to/fusion_det_100m_checkpoint.pth \
  --save_dir /root/autodl-tmp/cache/fusion_voxelnext_100m
```

4. 训练 tracker

```bash
python tools/train_tracker.py \
  --cfg_file tools/cfgs/tracking_models/track_mamba_100m.yaml \
  --extra_tag track_mamba_100m \
  --set DATA_CONFIG.ROOT_DIR /root/autodl-tmp/data/dair_v2x_tracking DATA_CONFIG.CACHE_DIR /root/autodl-tmp/cache/fusion_voxelnext_100m
```

### 200m

1. 训练检测

```bash
python tools/train.py \
  --cfg_file tools/cfgs/dair_v2x_models/fusion_voxelnext_200m.yaml \
  --extra_tag fusion_det_200m
```

2. 生成 detector cache

```bash
python tools/generate_tracking_cache.py \
  --detector_cfg tools/cfgs/dair_v2x_models/fusion_voxelnext_200m.yaml \
  --data_cfg tools/cfgs/dataset_configs/dair_v2x_tracking_dataset_200m.yaml \
  --ckpt /path/to/fusion_det_200m_checkpoint.pth \
  --save_dir /root/autodl-tmp/cache/fusion_voxelnext_200m
```

3. 训练 tracker

```bash
python tools/train_tracker.py \
  --cfg_file tools/cfgs/tracking_models/track_mamba_200m.yaml \
  --extra_tag track_mamba_200m \
  --set DATA_CONFIG.ROOT_DIR /root/autodl-tmp/data/dair_v2x_tracking DATA_CONFIG.CACHE_DIR /root/autodl-tmp/cache/fusion_voxelnext_200m
```

## 9. 新手最容易踩的坑

- 不要把 `output/`、`data/`、`cache/`、`*.pth` 提交到 git。
- 每次 `git add .` 之前先看一次 `git status`。
- AutoDL 开始训练前先执行 `git pull`。
- 不要在两边同时改同一个文件，否则容易产生冲突。
- 训练日志和 checkpoint 很大，建议只存在 AutoDL，不要回传到本地仓库。

## 10. 最短操作版

如果你只想先跑通一次，按这几步：

1. 本地：

```bash
cd /home/lx/sparseMamba
git init
git add .
git commit -m "init project"
git remote add origin https://gitee.com/your_name/sparseMamba.git
git branch -M main
git push -u origin main
```

2. AutoDL：

```bash
cd /root/autodl-tmp
git clone https://gitee.com/your_name/sparseMamba.git
cd sparseMamba
```

3. 配环境并训练。
