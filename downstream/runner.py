import os
import sys
import math
import glob
import uuid
import shutil
import random
import tempfile
import importlib
from pathlib import Path

import torch
import torchaudio
import numpy as np
from tqdm import tqdm
from tensorboardX import SummaryWriter
from torch.utils.data import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import is_initialized, get_rank, get_world_size

from s3prl import hub
from s3prl.optimizers import get_optimizer
from s3prl.schedulers import get_scheduler
from s3prl.upstream.interfaces import Featurizer
from s3prl.utility.helper import is_leader_process, get_model_state, show, defaultdict
from diarizen.models.eend.model_wavlm_conformer_gcc_solo_cspb import (
    Model as encoder_model,
)

import paderbox as pb
from diarizen.spatial_features.gcc_phat import get_gcc_for_all_channel_pairs_torch

# from huggingface_hub import HfApi, HfFolder, Repository

SAMPLE_RATE = 16000

MODEL_CARD_MARKDOWN = """---
datasets:
- superb
tags:
- library:s3prl
- benchmark:superb
- type:model
---

# Fine-tuned s3prl model

Upstream Model: {upstream_model}

## Model description

[More information needed]

## Intended uses & limitations

[More information needed]

## How to use

[More information needed]

## Limitations and bias

[More information needed]

## Training data

[More information needed]

## Training procedure

[More information needed]

## Evaluation results

[More information needed]

"""


class ModelEntry:
    def __init__(self, model, name, trainable, interfaces):
        self.model = model
        self.name = name
        self.trainable = trainable
        self.interfaces = interfaces


class Runner:
    """
    Used to handle high-level concepts of a ML experiment
    eg. training loop, evaluation loop, upstream propagation, optimization, logging, checkpoint saving
    """

    def __init__(self, args, config):
        self.args = args
        self.config = config
        self.init_ckpt = (
            torch.load(
                self.args.init_ckpt,
                map_location="cpu",
                weights_only=False,
            )
            if self.args.init_ckpt
            else {}
        )

        self.condition = self.config["downstream_expert"].get("condition", False)
        self.encoder_model = self.config["downstream_expert"].get(
            "encoder_model", False
        )
        self.spatial_conditioned = self.config["downstream_expert"].get(
            "spatial_conditioned", False
        )
        print(
            "[Runner] - Encoder model is {}".format(
                "enabled" if self.encoder_model else "disabled"
            )
        )
        model_path = self.config["downstream_expert"].get("condition_model_path", None)
        print(
            f"[Runner] - Conditioning model is {'enabled' if self.condition else 'disabled'}"
        )

        self.upstream = self._get_upstream()  # get e.g. wavlm
        self.featurizer = self._get_featurizer()  #  get feature extractor for wavlm?
        self.downstream = self._get_downstream()

        if self.condition:
            self.conditioning = self._get_conditioning(
                model_path=model_path
            )  # get conditioning model if any

            self.all_entries = [
                self.upstream,
                self.featurizer,
                self.conditioning,
                self.downstream,
            ]
        elif self.encoder_model:
            self.all_entries = [
                self.downstream,
            ]
        else:
            self.all_entries = [
                self.upstream,
                self.featurizer,
                self.downstream,
            ]

    def _load_weight(self, model, name):
        init_weight = self.init_ckpt.get(name)
        if init_weight:
            show(f"[Runner] - Loading {name} weights from the previous experiment")
            model.load_state_dict(init_weight)

    def _init_model(self, model, name, trainable, interfaces=None):
        for interface in interfaces or []:
            assert hasattr(model, interface), interface

        self._load_weight(model, name)

        if (
            is_initialized()
            and trainable
            and any((p.requires_grad for p in model.parameters()))
        ):
            model = DDP(
                model, device_ids=[self.args.local_rank], find_unused_parameters=True
            )
            for interface in interfaces or []:
                setattr(model, interface, getattr(model.module, interface))

        return ModelEntry(model, name, trainable, interfaces)

    def _get_upstream(self):
        if "from_hf_hub" in self.args and self.args.from_hf_hub == True:
            from huggingface_hub import snapshot_download

            print(
                f"[Runner] - Downloading upstream model {self.args.upstream} from the Hugging Face Hub"
            )
            filepath = snapshot_download(
                self.args.upstream, self.args.upstream_revision, use_auth_token=True
            )
            sys.path.append(filepath)

            dependencies = (Path(filepath) / "requirements.txt").resolve()
            print(
                "[Dependency] - The downloaded upstream model requires the following dependencies. Please make sure they are installed:"
            )
            for idx, line in enumerate(
                (Path(filepath) / "requirements.txt").open().readlines()
            ):
                print(f"{idx}. {line.strip()}")
            print(f"You can install them by:")
            print()
            print(f"pip install -r {dependencies}")
            print()

            from expert import UpstreamExpert

            Upstream = UpstreamExpert
            ckpt_path = os.path.join(filepath, self.args.upstream_model_name)
        else:
            Upstream = getattr(hub, self.args.upstream)
            ckpt_path = self.args.upstream_ckpt
        upstream_refresh = self.args.upstream_refresh

        if is_initialized() and get_rank() > 0:
            torch.distributed.barrier()
            upstream_refresh = False

        model = Upstream(
            ckpt=ckpt_path,
            model_config=self.args.upstream_model_config,
            refresh=upstream_refresh,
        ).to(self.args.device)

        if is_initialized() and get_rank() == 0:
            torch.distributed.barrier()

        return self._init_model(
            model=model,
            name="Upstream",
            trainable=self.args.upstream_trainable,
            interfaces=["get_downsample_rates"],
        )

    def _get_featurizer(self):
        model = Featurizer(
            upstream=self.upstream.model,
            feature_selection=self.args.upstream_feature_selection,
            layer_selection=self.args.upstream_layer_selection,
            upstream_device=self.args.device,
            normalize=self.args.upstream_feature_normalize,
        ).to(self.args.device)

        return self._init_model(
            model=model,
            name="Featurizer",
            trainable=True,
            interfaces=["output_dim", "downsample_rate"],
        )

    def _get_conditioning(self, model_path=None):
        # Placeholder for conditioning model initialization
        model = encoder_model(
            ffn=1024,
            num_layer_aux=5,
            max_num_spk=4,
            sin_cos=True,
            attention_in_aux=216,  # 2*513 + 0 0 für /4
            attention_in=256,
            ffn_hidden=1024,
            num_head=4,
            num_layer=4,
            dropout=0.1,
            chunk_size=8,
            use_posi=False,
            output_activate_function=False,
            max_speakers_per_chunk=4,
            selected_channel=0,
        )  # Replace with actual conditioning model initialization

        state_dict = torch.load(
            model_path,
            map_location="cpu",
            weights_only=True,
        )
        model.load_state_dict(state_dict)
        model = model.to(self.args.device)

        return self._init_model(
            model=model,
            name="Conditioning",
            trainable=True,
        )

    def _get_downstream(self):
        if not self.encoder_model and not self.spatial_conditioned:
            expert = importlib.import_module(
                f"s3prl.downstream.{self.args.downstream}.expert"
            )
        elif self.encoder_model or self.spatial_conditioned:
            expert = importlib.import_module(
                f"s3prl.downstream.{self.args.downstream}.expert_encoder"
            )
        Downstream = getattr(expert, "DownstreamExpert")

        model = Downstream(
            upstream_dim=self.featurizer.model.output_dim,
            upstream_rate=self.featurizer.model.downsample_rate,
            **self.config,
            **vars(self.args),
        ).to(self.args.device)

        return self._init_model(
            model=model,
            name="Downstream",
            trainable=True,
            interfaces=["get_dataloader", "log_records"],
        )

    def _get_optimizer(self, model_params):
        optimizer = get_optimizer(
            model_params, self.config["runner"]["total_steps"], self.config["optimizer"]
        )
        self._load_weight(optimizer, "Optimizer")
        return optimizer

    def _get_scheduler(self, optimizer):
        scheduler = get_scheduler(
            optimizer, self.config["runner"]["total_steps"], self.config["scheduler"]
        )
        self._load_weight(scheduler, "Scheduler")
        return scheduler

    def _create_model_card(self, path):
        model_card = MODEL_CARD_MARKDOWN.format(upstream_model=self.args.upstream)
        with open(os.path.join(path, "README.md"), "w") as f:
            f.write(model_card)

    def train(self):
        # trainable parameters and train/eval mode
        trainable_models = []
        trainable_paras = []
        # Adding models
        for entry in self.all_entries:
            if entry.trainable:
                print(f"[Runner] - {entry.name} is trainable")
                entry.model.train().to(self.args.device)
                trainable_models.append(entry.model)
                trainable_paras += list(entry.model.parameters())
            else:
                print(f"[Runner] - {entry.name} is frozen")
                entry.model.eval()

        # set amp
        amp = self.config["runner"].get("fp16", False)
        if amp:
            print("[Runner] - Enabled fp16 training")
            scaler = torch.cuda.amp.GradScaler()

        # optimizer
        optimizer = self._get_optimizer(trainable_models)

        # scheduler
        scheduler = None
        if self.config.get("scheduler"):
            scheduler = self._get_scheduler(optimizer)

        # specaug
        specaug = None
        if self.config.get("specaug"):
            from .specaug import SpecAug

            specaug = SpecAug(**self.config["specaug"])

        # progress bar
        tqdm_file = sys.stderr if is_leader_process() else open(os.devnull, "w")
        pbar = tqdm(
            total=self.config["runner"]["total_steps"],
            dynamic_ncols=True,
            desc="overall",
            file=tqdm_file,
        )
        init_step = self.init_ckpt.get("Step")
        if init_step:
            pbar.n = init_step

        # Tensorboard logging
        if is_leader_process():
            logger = SummaryWriter(self.args.expdir)

        batch_ids = []
        backward_steps = 0
        records = defaultdict(list)
        epoch = self.init_ckpt.get("Epoch", 0)
        train_split = self.config["runner"].get("train_dataloader", "train")
        # torch.cuda.memory._record_memory_history(max_entries=100000)

        while pbar.n < pbar.total:
            try:
                # TODO: Data loading depends on the downstream model, so we need to handle it carefully
                dataloader = self.downstream.model.get_dataloader(
                    train_split, epoch=epoch
                )
            except TypeError as e:
                if "unexpected keyword argument 'epoch'" in str(e):
                    dataloader = self.downstream.model.get_dataloader(train_split)
                    if hasattr(dataloader, "sampler") and isinstance(
                        dataloader.sampler, DistributedSampler
                    ):
                        dataloader.sampler.set_epoch(epoch)
                else:
                    raise

            for batch_id, (wavs, *others) in enumerate(
                tqdm(dataloader, dynamic_ncols=True, desc="train", file=tqdm_file)
            ):
                if batch_id == 0:
                    print(
                        f"[Runner] - Batch {batch_id} loaded with {len(wavs)} wavs and {len(others)} others"
                    )
                    for i, other in enumerate(others):
                        print(
                            f"  - Other {i}: {type(other[i])} with shape {other[i].shape if hasattr(other[i], 'shape') else 'N/A'}"
                        )

                # try/except block for forward/backward
                try:
                    if pbar.n >= pbar.total:
                        break
                    global_step = pbar.n + 1
                    # TODO: Get gcc features from wav or in dataset
                    wavs = [torch.FloatTensor(wav).to(self.args.device) for wav in wavs]
                    wavlm_wavs = [wav[:, 0] if wav.dim() == 2 else wav for wav in wavs]

                    with torch.cuda.amp.autocast(enabled=amp):
                        # TODO: Using upstream not possible when training encoder?
                        if not self.encoder_model:
                            if self.upstream.trainable:
                                features = self.upstream.model(wavlm_wavs)
                            else:
                                with torch.no_grad():
                                    features = self.upstream.model(wavlm_wavs)
                            # refines features, e.g. by selecting layers, normalizing, etc.
                            features = self.featurizer.model(wavlm_wavs, features)

                            if specaug:
                                features, _ = specaug(features)

                        # try:
                        #     torch.cuda.memory._dump_snapshot(
                        #         f"/net/vol/deegen/tmp/{batch_id}.pickle"
                        #     )
                        # except Exception as e:
                        #     logger.error(f"Failed to capture memory snapshot {e}")

                        # Option for encoder model or conditioning system
                        if self.condition or self.encoder_model:
                            spatial_features = others[-1]
                            # spatial_features = torch.stack(
                            #     [x.to(torch.complex64) for x in spatial_features],
                            #     dim=0,
                            # ).to(self.args.device)
                            spatial_features = [
                                x.to(self.args.device) for x in spatial_features
                            ]

                            if self.condition:
                                cond = self.conditioning.model(
                                    wavs, gcc_features=spatial_features
                                )
                                loss = self.downstream.model(
                                    train_split,
                                    features,
                                    labels=others[0],
                                    lengths=others[1],
                                    rec_id=others[2],
                                    cond=cond,
                                    records=records,
                                )
                            elif self.encoder_model:
                                # prediction = self.encoder.model(
                                loss = self.downstream.model(
                                    train_split,
                                    features=spatial_features,
                                    labels=others[0],
                                    # lengths=others[1],
                                    filenames=others[1],
                                    records=records,
                                    # gcc_features=spatial_features,
                                )

                            # try:
                            #     torch.cuda.memory._dump_snapshot(
                            #         f"/net/vol/deegen/tmp/{batch_id}_222.pickle"
                            #     )
                            # except Exception as e:
                            #     logger.error(f"Failed to capture memory snapshot {e}")
                        else:
                            if self.spatial_conditioned:
                                loss = self.downstream.model(
                                    train_split,
                                    features,
                                    labels=others[0],
                                    filenames=others[1],
                                    spatial_features=[
                                        x.to(self.args.device) for x in others[2]
                                    ],
                                    records=records,
                                )
                            else:
                                loss = self.downstream.model(
                                    train_split,
                                    features,
                                    *others,
                                    records=records,
                                )

                        # Stop recording memory snapshot history.
                        # torch.cuda.memory._record_memory_history(enabled=None)
                    batch_ids.append(batch_id)

                    gradient_accumulate_steps = self.config["runner"].get(
                        "gradient_accumulate_steps"
                    )
                    loss = loss / gradient_accumulate_steps
                    if amp:
                        scaler.scale(loss).backward()
                    else:
                        loss.backward()
                    del loss

                except RuntimeError as e:
                    if "CUDA out of memory" in str(e):
                        print(f"[Runner] - CUDA out of memory at step {global_step}")
                        if is_initialized():
                            raise
                        with torch.cuda.device(self.args.device):
                            torch.cuda.empty_cache()
                        optimizer.zero_grad()
                        continue
                    else:
                        raise

                # whether to accumulate gradient
                backward_steps += 1
                if backward_steps % gradient_accumulate_steps > 0:
                    continue

                # unscale
                if amp:
                    scaler.unscale_(optimizer)

                # gradient clipping
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    trainable_paras, self.config["runner"]["gradient_clipping"]
                )

                # optimize
                if amp:
                    scaler.step(optimizer)
                    scaler.update()
                elif math.isnan(grad_norm):
                    print(f"[Runner] - grad norm is NaN at step {global_step}")
                else:
                    optimizer.step()
                optimizer.zero_grad()

                # adjust learning rate
                if scheduler:
                    scheduler.step()

                # evaluation and save checkpoint
                save_names = []
                eval_step = global_step % self.config["runner"]["eval_step"] == 0

                # # # TODO: REMOVE
                # eval_step = True  # For debugging

                if eval_step and torch.distributed.is_initialized():
                    torch.distributed.barrier()

                if eval_step and is_leader_process():
                    for split in self.config["runner"]["eval_dataloaders"]:
                        save_names += self.evaluate(split, logger, global_step)

                if eval_step and torch.distributed.is_initialized():
                    torch.distributed.barrier()

                if not is_leader_process():
                    batch_ids = []
                    records = defaultdict(list)
                    pbar.update(1)
                    continue

                # logging
                if global_step % self.config["runner"]["log_step"] == 0:
                    # TODO: uses downstream model
                    self.downstream.model.log_records(
                        train_split,
                        records=records,
                        logger=logger,
                        global_step=global_step,
                        batch_ids=batch_ids,
                        total_batch_num=len(dataloader),
                    )
                    batch_ids = []
                    records = defaultdict(list)

                if global_step % self.config["runner"]["save_step"] == 0:

                    def check_ckpt_num(directory):
                        max_keep = self.config["runner"]["max_keep"]
                        ckpt_pths = glob.glob(f"{directory}/states-*.ckpt")
                        if len(ckpt_pths) >= max_keep:
                            ckpt_pths = sorted(
                                ckpt_pths,
                                key=lambda pth: int(pth.split("-")[-1].split(".")[0]),
                            )
                            for ckpt_pth in ckpt_pths[: len(ckpt_pths) - max_keep + 1]:
                                os.remove(ckpt_pth)

                    check_ckpt_num(self.args.expdir)
                    save_names.append(f"states-{global_step}.ckpt")

                if len(save_names) > 0:
                    all_states = {
                        "Optimizer": optimizer.state_dict(),
                        "Step": global_step,
                        "Epoch": epoch,
                        "Args": self.args,
                        "Config": self.config,
                    }

                    for entry in self.all_entries:
                        if entry.trainable:
                            all_states[entry.name] = get_model_state(entry.model)

                    if scheduler:
                        all_states["Scheduler"] = scheduler.state_dict()

                    if is_initialized():
                        all_states["WorldSize"] = get_world_size()

                    save_paths = [
                        os.path.join(self.args.expdir, name) for name in save_names
                    ]
                    tqdm.write(f"[Runner] - Save the checkpoint to:")
                    for i, path in enumerate(save_paths):
                        tqdm.write(f"{i + 1}. {path}")
                        torch.save(all_states, path)

                pbar.update(1)
            epoch += 1

        pbar.close()

        if self.args.push_to_hf_hub:
            self.push_to_huggingface_hub()
        if is_leader_process():
            logger.close()

    def evaluate(self, split=None, logger=None, global_step=0):
        """evaluate function will always be called on a single process even during distributed training"""

        # When this member function is called directly by command line
        not_during_training = split is None and logger is None and global_step == 0
        if not_during_training:
            split = self.args.evaluate_split
            tempdir = tempfile.mkdtemp()
            logger = SummaryWriter(tempdir)

        # fix seed to guarantee the same evaluation protocol across steps
        random.seed(self.args.seed)
        np.random.seed(self.args.seed)
        torch.manual_seed(self.args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.args.seed)
            with torch.cuda.device(self.args.device):
                torch.cuda.empty_cache()

        # record original train/eval states and set all models to eval
        trainings = []
        for entry in self.all_entries:
            trainings.append(entry.model.training)
            entry.model.eval()

        # Evaluate the underlying module on rank 0. Calling a DDP wrapper from
        # only one rank can introduce buffer-synchronization collectives that
        # conflict with the barriers used by the other ranks.
        downstream_model = (
            self.downstream.model.module
            if isinstance(self.downstream.model, DDP)
            else self.downstream.model
        )
        upstream_model = (
            self.upstream.model.module
            if isinstance(self.upstream.model, DDP)
            else self.upstream.model
        )

        featurizer_model = (
            self.featurizer.model.module
            if isinstance(self.featurizer.model, DDP)
            else self.featurizer.model
        )
        # prepare data
        dataloader = downstream_model.get_dataloader(split)
        evaluate_ratio = float(self.config["runner"].get("evaluate_ratio", 1))
        evaluate_steps = round(len(dataloader) * evaluate_ratio)

        batch_ids = []
        records = defaultdict(list)
        for batch_id, (wavs, *others) in enumerate(
            tqdm(dataloader, dynamic_ncols=True, desc=split, total=evaluate_steps)
        ):
            # TODO: Eval steps adapt when train dataset larger
            if batch_id > evaluate_steps:
                break

            wavs = [torch.FloatTensor(wav).to(self.args.device) for wav in wavs]
            wavlm_wavs = [wav[:, 0] if wav.dim() == 2 else wav for wav in wavs]
            with torch.no_grad():
                if not self.encoder_model:
                    try:
                        features = upstream_model(wavlm_wavs)
                    except RuntimeError:
                        print("Wav len", len(wavlm_wavs), "padding...", flush=True)
                        wavlm_wavs[0] = torch.nn.functional.pad(
                            wavlm_wavs[0],
                            (0, 320),
                            mode="constant",
                            value=0,
                        )
                        features = upstream_model(wavlm_wavs)

                    features = featurizer_model(
                        wavlm_wavs,
                        features,
                    )
                    if hasattr(downstream_model, "spatial_encoder"):
                        downstream_model(
                            split,
                            features,
                            labels=others[0],
                            filenames=others[1],
                            spatial_features=[
                                x.to(self.args.device) for x in others[2]
                            ],
                            records=records,
                            batch_id=batch_id,
                        )
                    else:
                        downstream_model(
                            split,
                            features,
                            *others,
                            records=records,
                            batch_id=batch_id,
                        )
                elif self.encoder_model:
                    spatial_features = others[-1]
                    spatial_features = [
                        x.to(self.args.device) for x in spatial_features
                    ]
                    try:
                        downstream_model(
                            split,
                            features=spatial_features,
                            labels=others[0],
                            filenames=others[1],
                            records=records,
                            batch_id=batch_id,
                        )
                    except RuntimeError as e:
                        # some segmenst are too short and kernel size error would be creadted by ipstream model
                        print(
                            "spatial_features len",
                            len(spatial_features),
                            "skipping...",
                            flush=True,
                        )  # , np.array(wavs).shape)
                        # spatial_features[0] = torch.nn.functional.pad(
                        #     spatial_features[0],
                        #     (0, 0, 0, 0, 0, 1, 0, 0),
                        #     mode="constant",
                        #     value=0,
                        # )
                        continue
                        self.downstream.model(
                            split,
                            features=spatial_features,
                            labels=others[0],
                            filenames=others[1],
                            records=records,
                            batch_id=batch_id,
                        )
                    # self.downstream.model(
                    #     split,
                    #     features=spatial_features,
                    #     labels=others[0],
                    #     filenames=others[1],
                    #     records=records,
                    #     batch_id=batch_id,
                    # )
                batch_ids.append(batch_id)

        #            with torch.no_grad():
        #                features = self.upstream.model(wavs)
        #                features = self.featurizer.model(wavs, features)
        #                self.downstream.model(
        #                    split,
        #                    features,
        #                    *others,
        #                    records=records,
        #                    batch_id=batch_id,
        #                )
        #                batch_ids.append(batch_id)

        save_names = downstream_model.log_records(
            split,
            records=records,
            logger=logger,
            global_step=global_step,
            batch_ids=batch_ids,
            total_batch_num=len(dataloader),
        )
        batch_ids = []
        records = defaultdict(list)

        # prepare back to training
        if torch.cuda.is_available():
            with torch.cuda.device(self.args.device):
                torch.cuda.empty_cache()

        for entry, training in zip(self.all_entries, trainings):
            if training:
                entry.model.train().to(self.args.device)

        if not_during_training:
            logger.close()
            shutil.rmtree(tempdir)

        return [] if type(save_names) is not list else save_names

    def inference(self):
        filepath = Path(self.args.evaluate_split)
        assert filepath.is_file(), filepath
        filename = filepath.stem

        if hasattr(self.downstream.model, "load_audio"):
            wav = self.downstream.model.load_audio(filepath)
        else:
            wav, sr = torchaudio.load(str(filepath))
            assert sr == SAMPLE_RATE, sr
        if self.encoder_model or hasattr(self.downstream.model, "spatial_encoder"):
            spatial_features = self.compute_gcc_features(
                wav.transpose(1, 0)
            )  # (channels, samples)
            spatial_features = spatial_features.to(torch.complex64)
            spatial_features = [x.to(self.args.device) for x in spatial_features]

        else:
            wavs = [wav.view(-1).to(self.args.device)]

        for entry in self.all_entries:
            entry.model.eval()

        with torch.no_grad():
            if not self.encoder_model:
                features = self.upstream.model(wavs)
                features = self.featurizer.model(wavs, features)
                if self.spatial_conditioned:
                    self.downstream.model.inference(
                        features, [filename], spatial_features
                    )
                else:
                    self.downstream.model.inference(features, [filename])

            elif self.encoder_model:
                self.downstream.model.inference(spatial_features, [filename])

    def push_to_huggingface_hub(self):
        """Creates a downstream repository on the Hub and pushes training artifacts to it."""
        if self.args.hf_hub_org.lower() != "none":
            organization = self.args.hf_hub_org
        else:
            organization = os.environ.get("HF_USERNAME")
        huggingface_token = HfFolder.get_token()
        print(f"[Runner] - Organisation to push fine-tuned model to: {organization}")

        # Extract upstream repository metadata
        if self.args.hub == "huggingface":
            model_info = HfApi().model_info(self.args.upstream, token=huggingface_token)
            downstream_model_id = model_info.sha
            # Exclude "/" characters from downstream repo ID
            upstream_model_id = model_info.modelId.replace("/", "__")
        else:
            upstream_model_id = self.args.upstream.replace("/", "__")
            downstream_model_id = str(uuid.uuid4())[:8]
        repo_name = f"{upstream_model_id}__{downstream_model_id}"
        # Create downstream repo on the Hub
        repo_url = HfApi().create_repo(
            token=huggingface_token,
            name=repo_name,
            organization=organization,
            exist_ok=True,
            private=False,
        )
        print(f"[Runner] - Created Hub repo: {repo_url}")

        # Download repo
        HF_HUB_DIR = "hf_hub"
        REPO_ROOT_DIR = os.path.join(self.args.expdir, HF_HUB_DIR, repo_name)
        REPO_TASK_DIR = os.path.join(
            REPO_ROOT_DIR, self.args.downstream, self.args.expname
        )
        print(f"[Runner] - Cloning Hub repo to {REPO_ROOT_DIR}")
        model_repo = Repository(
            local_dir=REPO_ROOT_DIR,
            clone_from=repo_url,
            use_auth_token=huggingface_token,
        )
        # Pull latest changes if they exist
        model_repo.git_pull()

        # Copy checkpoints, tensorboard logs, and args / configs
        # Note that this copies all files from the experiment directory,
        # including those from multiple runs
        shutil.copytree(
            self.args.expdir,
            REPO_TASK_DIR,
            dirs_exist_ok=True,
            ignore=shutil.ignore_patterns(HF_HUB_DIR),
        )

        # By default we use model.ckpt in the PreTrainedModel interface, so
        # rename the best checkpoint to match this convention
        checkpoints = list(Path(REPO_TASK_DIR).glob("*best*.ckpt"))
        if len(checkpoints) == 0:
            print(
                "[Runner] - Did not find a best checkpoint! Using the final checkpoint instead ..."
            )
            CKPT_PATH = os.path.join(
                REPO_TASK_DIR, f"states-{self.config['runner']['total_steps']}.ckpt"
            )
        elif len(checkpoints) > 1:
            print(
                f"[Runner] - More than one best checkpoint found! Using {checkpoints[0]} as default ..."
            )
            CKPT_PATH = checkpoints[0]
        else:
            print(f"[Runner] - Found best checkpoint {checkpoints[0]}!")
            CKPT_PATH = checkpoints[0]
        shutil.move(CKPT_PATH, os.path.join(REPO_TASK_DIR, "model.ckpt"))
        model_repo.lfs_track("*.ckpt")

        # Write model card
        self._create_model_card(REPO_ROOT_DIR)

        # Push everything to the Hub
        print("[Runner] - Pushing model files to the Hub ...")
        model_repo.push_to_hub()
        print("[Runner] - Training run complete!")

    def compute_gcc_features(self, audio):

        apply_ifft = False
        fmin = 125
        fmax = 3500
        fft_size = 1024
        k_min = int(np.round(fmin / (self.sample_rate / 2) * (fft_size // 2 + 1)))
        k_max = int(np.round(fmax / (self.sample_rate / 2) * (fft_size // 2 + 1)))
        # gcc_features = gcc_features[:, :, k_min:k_max]   # 216 freq bins long

        data_pad = self.pad_data(audio, size=fft_size)
        # print(
        #     f"[ASRDataset] - Padded audio shape: {data_pad.shape} (original: {audio.shape})"
        # )
        sigs_stft = pb.transform.stft(
            data_pad[0], size=fft_size, shift=320, pad=False, fading=False
        )
        magnitude = torch.from_numpy(
            np.abs(sigs_stft)[:, k_min:k_max]
        )  # (frames, freq))
        gcc_features = self.compute_gcc(
            data_pad,
            frame_size_gcc=fft_size,
            frame_shift_gcc=320,
            f_max_gcc=fmax,
            f_min=fmin,
            apply_ifft=apply_ifft,
        )

        if magnitude is not None:
            gcc_features = torch.concat([gcc_features, magnitude[:, None, :]], dim=1)

        return gcc_features

    def compute_gcc(
        self,
        waveforms_mc,
        frame_size_gcc=4096,
        frame_shift_gcc=311,
        avg_len_gcc=4,
        search_range_gcc=10,
        f_max_gcc=None,
        f_min=125,
        apply_ifft=True,
    ):
        """
        Compute GCC features from multichannel waveforms.
        returns:
            batch_gcc_features: (batch, frame, channel, channel, search_range)
        """
        # TODO: try different stft values for better gcc but need fit frames of WAVLM
        sigs_stft = pb.transform.stft(
            waveforms_mc, frame_size_gcc, frame_shift_gcc, pad=False, fading=False
        )
        sigs_stft = torch.from_numpy(sigs_stft)  # (frames, channels, freq)
        gcc_features = get_gcc_for_all_channel_pairs_torch(
            sigs_stft, f_min=f_min, f_max=f_max_gcc, apply_ifft=apply_ifft
        )
        return gcc_features

    def get_num_frames(self, L, n_fft, hop):
        return math.floor((L - n_fft) / hop) + 1

    def pad_data(self, data, size=4096):
        L = data.shape[-1]
        n_fft_wavlm = 400
        hop = 320
        N_ref = self.get_num_frames(L, n_fft_wavlm, hop)

        n_fft_big = size
        N_big = self.get_num_frames(L, n_fft_big, hop)

        L_target = (N_ref - 1) * hop + n_fft_big
        pad = max(0, L_target - L)

        last_vals = data[:, -1:]  # Shape (C,1)
        pad_block = np.repeat(last_vals, pad, axis=1)  # Shape (C,pad)
        return np.concatenate([data, pad_block], axis=1)
