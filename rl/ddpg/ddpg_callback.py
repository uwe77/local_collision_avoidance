from stable_baselines3.common.callbacks import CheckpointCallback
import os
import shutil


class OverwriteCheckpointCallback(CheckpointCallback):
    def __init__(
        self,
        save_freq: int,
        save_path: str,
        name_prefix: str = "rl_model",
        save_replay_buffer: bool = False,
        save_vecnormalize: bool = False,
        overwrite: bool = False,
        verbose: int = 0,
    ):
        super().__init__(
            save_freq=save_freq,
            save_path=save_path,
            name_prefix=name_prefix,
            save_replay_buffer=save_replay_buffer,
            save_vecnormalize=save_vecnormalize,
            verbose=verbose,
        )
        self.overwrite = overwrite

        if self.overwrite:
            # Final file path instead of directory
            self.save_file = os.path.join(save_path, f"{name_prefix}.zip")
            self.replay_buffer_path = self.save_file.replace(".zip", "_replay_buffer.pkl")
            self.vecnormalize_path = self.save_file.replace(".zip", "_vecnormalize.pkl")

            # 🧹 Remove existing folder if it exists
            conflicting_dir = os.path.join(save_path, name_prefix)
            if os.path.isdir(conflicting_dir):
                shutil.rmtree(conflicting_dir)

    def _on_step(self) -> bool:
        if self.n_calls % self.save_freq == 0:
            if self.overwrite:
                self._save_overwrite()
            else:
                super()._on_step()
        return True

    def _save_overwrite(self):
        if self.verbose > 0:
            print(f"Overwriting checkpoint to {self.save_file}")
        self.model.save(self.save_file)

        if self.save_replay_buffer and self.model.replay_buffer is not None:
            self.model.save_replay_buffer(self.replay_buffer_path)

        if self.save_vecnormalize and hasattr(self.model, "get_vec_normalize_env"):
            vec_normalize_env = self.model.get_vec_normalize_env()
            if vec_normalize_env is not None:
                vec_normalize_env.save(self.vecnormalize_path)

