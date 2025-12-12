from typing import Optional, Literal
import masknmf
from masknmf.compression import PMDArray
from masknmf.compression.denoising import train_total_variance_denoiser
from masknmf.compression.decomposition import pmd_decomposition
from masknmf import ArrayLike
import numpy as np
from masknmf.compression.spatial_denoiser import train_spatial_denoiser, create_pmd_denoiser

class CompressStrategy:

    def __init__(self,
                 dataset: ArrayLike | None,
                 block_sizes: tuple[int, int] = (32, 32),
                 frame_range: int | None  = None,
                 max_components: int = 20,
                 sim_conf: int = 5,
                 frame_batch_size: int = 10000,
                 max_consecutive_failures: int=1,
                 spatial_avg_factor:int=1,
                 temporal_avg_factor:int=1,
                 compute_normalizer: Optional[bool] = True,
                 pixel_weighting: Optional[np.ndarray] = None,
                 device: Literal["auto", "cpu", "cuda"] = "auto",
                 ):

        self._dataset = dataset

        ##User-settable parameters
        self._block_sizes = block_sizes
        self._frame_range = frame_range
        self._max_components = max_components
        self._frame_batch_size = frame_batch_size
        self._spatial_avg_factor = spatial_avg_factor
        self._temporal_avg_factor = temporal_avg_factor
        self._device=device

        ##Non user-settable parameters
        self._sim_conf = sim_conf
        self._max_consecutive_failures=max_consecutive_failures
        self._compute_normalizer = compute_normalizer
        self._pixel_weighting = pixel_weighting

        self._results = None

    @property
    def dataset(self) -> ArrayLike | None:
        return self._dataset

    @dataset.setter
    def dataset(self, new_dataset: ArrayLike):
        self._dataset = new_dataset

    @property
    def block_sizes(self) -> tuple[int, int]:
        return self._block_sizes

    @block_sizes.setter
    def block_sizes(self, new_sizes: tuple[int, int]):
        self._block_sizes = new_sizes

    @property
    def frame_range(self) -> int | None:
        return self._frame_range

    @frame_range.setter
    def frame_range(self, new_frame_range: int):
        self._frame_range = new_frame_range

    @property
    def max_consecutive_failures(self) -> int:
        return self._max_consecutive_failures

    @max_consecutive_failures.setter
    def max_consecutive_failures(self, new_num: int):
        self._max_consecutive_failures = new_num

    @property
    def sim_conf(self) -> int:
        return self._sim_conf

    @sim_conf.setter
    def sim_conf(self, new_sim_conf: int):
        self._sim_conf = new_sim_conf

    @property
    def max_components(self):
        return self._max_components

    @max_components.setter
    def max_components(self, num_comps: int):
        self._max_components = num_comps

    @property
    def frame_batch_size(self) -> int:
        return self._frame_batch_size

    @frame_batch_size.setter
    def frame_batch_size(self, new_batch_size:int):
        self._frame_batch_size = new_batch_size

    @property
    def spatial_avg_factor(self) -> int:
        return self._spatial_avg_factor

    @spatial_avg_factor.setter
    def spatial_avg_factor(self, new_spatial_avg_factor: int):
        self._spatial_avg_factor = new_spatial_avg_factor

    @property
    def temporal_avg_factor(self) -> int:
        return self._temporal_avg_factor

    @temporal_avg_factor.setter
    def temporal_avg_factor(self, new_temporal_avg_factor: int):
        self._temporal_avg_factor = new_temporal_avg_factor

    @property
    def device(self) ->str:
        return self._device

    @device.setter
    def device(self, new_device: str):
        self._device = new_device

    @property
    def results(self) -> PMDArray | None:
        return self._results

    def compress(self) -> PMDArray:
        if self.dataset is None:
            raise ValueError("No dataset provided. "
                             "Provide a dataset at construction time of the"
                             " strategy object or set the dataset property")
        self._results = pmd_decomposition(self.dataset,
                                          self.block_sizes,
                                          frame_range=self.frame_range,
                                          max_components=self.max_components,
                                          sim_conf=self._sim_conf,
                                          frame_batch_size=self.frame_batch_size,
                                          max_consecutive_failures=self._max_consecutive_failures,
                                          spatial_avg_factor=self.spatial_avg_factor,
                                          temporal_avg_factor=self.temporal_avg_factor,
                                          compute_normalizer=self._compute_normalizer,
                                          pixel_weighting=self._pixel_weighting,
                                          device=self.device)

        return self._results

class CompressTemporalDenoiseStrategy(CompressStrategy):


    def __init__(self,
                 dataset: ArrayLike | None,
                 block_sizes: tuple[int, int] = (32, 32),
                 frame_range: int | None  = None,
                 max_components: int = 20,
                 sim_conf: int = 5,
                 frame_batch_size: int = 10000,
                 max_consecutive_failures: int=1,
                 spatial_avg_factor:int=1,
                 temporal_avg_factor:int=1,
                 compute_normalizer: Optional[bool] = True,
                 pixel_weighting: Optional[np.ndarray] = None,
                 device: Literal["auto", "cpu", "cuda"] = "auto",
                 noise_variance_quantile: float = 0.7,
                 num_epochs: int = 5
                 ):

        super().__init__(dataset,
                         block_sizes,
                         frame_range,
                         max_components,
                         sim_conf,
                         frame_batch_size,
                         max_consecutive_failures,
                         spatial_avg_factor,
                         temporal_avg_factor,
                         compute_normalizer,
                         pixel_weighting,
                         device)
        self._num_epochs = num_epochs
        self._noise_variance_quantile = noise_variance_quantile

    @property
    def num_epochs(self) -> int:
        return self._num_epochs

    @num_epochs.setter
    def num_epochs(self, new_num_epochs: int):
        self._num_epochs = new_num_epochs

    @property
    def noise_variance_quantile(self) -> float:
        return self._noise_variance_quantile

    @noise_variance_quantile.setter
    def noise_variance_quantile(self, new_noise_variance_quantile: float):
        self._noise_variance_quantile = new_noise_variance_quantile

    def compress(self):

        if self.dataset is None:
            raise ValueError("No dataset provided. "
                             "Provide a dataset at construction time of the"
                             " strategy object or set the dataset property")
        pmd_no_denoiser = pmd_decomposition(self.dataset,
                                            self.block_sizes,
                                            frame_range=self.frame_range,
                                            max_components=self.max_components,
                                            sim_conf=self._sim_conf,
                                            frame_batch_size=self.frame_batch_size,
                                            max_consecutive_failures=self._max_consecutive_failures,
                                            spatial_avg_factor=self.spatial_avg_factor,
                                            temporal_avg_factor=self.temporal_avg_factor,
                                            compute_normalizer=self._compute_normalizer,
                                            pixel_weighting=self._pixel_weighting,
                                            device=self.device)

        v = pmd_no_denoiser.v.cpu()
        trained_model, _ = masknmf.compression.denoising.train_total_variance_denoiser(v,
                                                                                       max_epochs=self.num_epochs,
                                                                                       batch_size=128,
                                                                                       learning_rate=1e-4)

        curr_temporal_denoiser = masknmf.compression.PMDTemporalDenoiser(trained_model, self.noise_variance_quantile)

        self._results = masknmf.compression.pmd_decomposition(self.dataset,
                                                             self.block_sizes,
                                                             frame_range=self.frame_range,
                                                             max_components=self.max_components,
                                                             sim_conf=self._sim_conf,
                                                             frame_batch_size=self.frame_batch_size,
                                                             max_consecutive_failures=self._max_consecutive_failures,
                                                             spatial_avg_factor=self.spatial_avg_factor,
                                                             temporal_avg_factor=self.temporal_avg_factor,
                                                             compute_normalizer=self._compute_normalizer,
                                                             pixel_weighting=self._pixel_weighting,
                                                             device=self.device,
                                                             temporal_denoiser=curr_temporal_denoiser)

        return self._results

class CompressSpatialDenoiseStrategy:
    """
    Compression strategy with spatial denoising.
    First runs PMD without denoiser, trains spatial denoiser on spatial components,
    then runs PMD again with the trained spatial denoiser.
    """

    def __init__(self,
                 dataset: ArrayLike | None,
                 block_sizes: tuple[int, int] = (32, 32),
                 frame_range: int | None = None,
                 max_components: int = 20,
                 sim_conf: int = 5,
                 frame_batch_size: int = 10000,
                 max_consecutive_failures: int = 1,
                 spatial_avg_factor: int = 1,
                 temporal_avg_factor: int = 1,
                 compute_normalizer: Optional[bool] = True,
                 pixel_weighting: Optional[np.ndarray] = None,
                 device: Literal["auto", "cpu", "cuda"] = "auto",
                 # Spatial denoiser parameters
                 spatial_denoiser_epochs: int = 5,
                 spatial_denoiser_batch_size: int = 32,
                 spatial_denoiser_lr: float = 1e-4,
                 noise_variance_quantile: float = 0.7,
                 denoiser_padding: int = 12,
                 patch_h: int = 40,
                 patch_w: int = 40,
                 ):

        self._dataset = dataset

        # User-settable PMD parameters
        self._block_sizes = block_sizes
        self._frame_range = frame_range
        self._max_components = max_components
        self._frame_batch_size = frame_batch_size
        self._spatial_avg_factor = spatial_avg_factor
        self._temporal_avg_factor = temporal_avg_factor
        self._device = device

        # Non user-settable PMD parameters
        self._sim_conf = sim_conf
        self._max_consecutive_failures = max_consecutive_failures
        self._compute_normalizer = compute_normalizer
        self._pixel_weighting = pixel_weighting

        # Spatial denoiser parameters
        self._spatial_denoiser_epochs = spatial_denoiser_epochs
        self._spatial_denoiser_batch_size = spatial_denoiser_batch_size
        self._spatial_denoiser_lr = spatial_denoiser_lr
        self._noise_variance_quantile = noise_variance_quantile
        self._denoiser_padding = denoiser_padding
        self._patch_h = patch_h
        self._patch_w = patch_w

        self._results = None

    # Properties for PMD parameters
    @property
    def dataset(self) -> ArrayLike | None:
        return self._dataset

    @dataset.setter
    def dataset(self, new_dataset: ArrayLike):
        self._dataset = new_dataset

    @property
    def block_sizes(self) -> tuple[int, int]:
        return self._block_sizes

    @block_sizes.setter
    def block_sizes(self, new_sizes: tuple[int, int]):
        self._block_sizes = new_sizes

    @property
    def frame_range(self) -> int | None:
        return self._frame_range

    @frame_range.setter
    def frame_range(self, new_frame_range: int):
        self._frame_range = new_frame_range

    @property
    def max_components(self):
        return self._max_components

    @max_components.setter
    def max_components(self, num_comps: int):
        self._max_components = num_comps

    @property
    def device(self) -> str:
        return self._device

    @device.setter
    def device(self, new_device: str):
        self._device = new_device

    @property
    def frame_batch_size(self) -> int:
        return self._frame_batch_size

    @frame_batch_size.setter
    def frame_batch_size(self, new_batch_size: int):
        self._frame_batch_size = new_batch_size

    @property
    def spatial_avg_factor(self) -> int:
        return self._spatial_avg_factor

    @spatial_avg_factor.setter
    def spatial_avg_factor(self, new_spatial_avg_factor: int):
        self._spatial_avg_factor = new_spatial_avg_factor

    @property
    def temporal_avg_factor(self) -> int:
        return self._temporal_avg_factor

    @temporal_avg_factor.setter
    def temporal_avg_factor(self, new_temporal_avg_factor: int):
        self._temporal_avg_factor = new_temporal_avg_factor

    # Properties for spatial denoiser parameters
    @property
    def spatial_denoiser_epochs(self) -> int:
        return self._spatial_denoiser_epochs

    @spatial_denoiser_epochs.setter
    def spatial_denoiser_epochs(self, new_epochs: int):
        self._spatial_denoiser_epochs = new_epochs

    @property
    def noise_variance_quantile(self) -> float:
        return self._noise_variance_quantile

    @noise_variance_quantile.setter
    def noise_variance_quantile(self, new_quantile: float):
        self._noise_variance_quantile = new_quantile

    @property
    def results(self) -> PMDArray | None:
        return self._results

    def compress(self) -> PMDArray:
        """
        Run compression with spatial denoising:
        1. First PMD decomposition without denoiser
        2. Extract spatial components and train spatial denoiser
        3. Second PMD decomposition with trained spatial denoiser
        """
        if self.dataset is None:
            raise ValueError("No dataset provided. "
                             "Provide a dataset at construction time or "
                             "set the dataset property")

        print("\n" + "="*60)
        print("STEP 1: Initial PMD decomposition (no denoiser)")
        print("="*60)
        
        # First pass: PMD without denoiser
        pmd_no_denoiser = pmd_decomposition(
            self.dataset,
            self.block_sizes,
            frame_range=self.frame_range,
            max_components=self.max_components,
            sim_conf=self._sim_conf,
            frame_batch_size=self.frame_batch_size,
            max_consecutive_failures=self._max_consecutive_failures,
            spatial_avg_factor=self.spatial_avg_factor,
            temporal_avg_factor=self.temporal_avg_factor,
            compute_normalizer=self._compute_normalizer,
            pixel_weighting=self._pixel_weighting,
            device=self.device
        )

        print(f"Initial PMD rank: {pmd_no_denoiser.pmd_rank}")

        print("\n" + "="*60)
        print("STEP 2: Training spatial denoiser")
        print("="*60)

        # Extract spatial components from U matrix
        u_dense = pmd_no_denoiser.u.to_dense()
        
        # Get spatial dimensions from dataset
        if hasattr(self.dataset, 'shape'):
            H, W = self.dataset.shape[1], self.dataset.shape[2]
        else:
            # Try to infer from first frame
            first_frame = self.dataset[0]
            H, W = first_frame.shape
        
        # Reshape U to (H, W, num_components) then to (num_components, H, W)
        u_reshaped = u_dense.reshape(H, W, -1)
        spatial_components = u_reshaped.permute(2, 0, 1)  # (rank, H, W)
        
        print(f"Spatial components shape: {spatial_components.shape}")

        # Train spatial denoiser
        denoiser_config = {
            'max_epochs': self._spatial_denoiser_epochs,
            'batch_size': self._spatial_denoiser_batch_size,
            'learning_rate': self._spatial_denoiser_lr,
            'patch_h': self._patch_h,
            'patch_w': self._patch_w,
            'device': self.device
        }

        trained_spatial_model, training_info = train_spatial_denoiser(
            spatial_components,
            config=denoiser_config,
            output_dir=None  # Set to a path if you want to save
        )

        # Create PMD-compatible denoiser
        spatial_denoiser = create_pmd_denoiser(
            trained_model=trained_spatial_model,
            noise_variance_quantile=self._noise_variance_quantile,
            padding=self._denoiser_padding,
            device=self.device
        )

        print("\n" + "="*60)
        print("STEP 3: Final PMD decomposition (with spatial denoiser)")
        print("="*60)

        # Second pass: PMD with trained spatial denoiser
        self._results = pmd_decomposition(
            self.dataset,
            self.block_sizes,
            frame_range=self.frame_range,
            max_components=self.max_components,
            sim_conf=self._sim_conf,
            frame_batch_size=self.frame_batch_size,
            max_consecutive_failures=self._max_consecutive_failures,
            spatial_avg_factor=self.spatial_avg_factor,
            temporal_avg_factor=self.temporal_avg_factor,
            compute_normalizer=self._compute_normalizer,
            pixel_weighting=self._pixel_weighting,
            device=self.device,
            spatial_denoiser=spatial_denoiser
        )

        print(f"\nFinal PMD rank (with spatial denoiser): {self._results.pmd_rank}")
        print(f"Rank change: {pmd_no_denoiser.pmd_rank} -> {self._results.pmd_rank}")
        print("="*60 + "\n")

        return self._results


class CompressSpatialTemporalDenoiseStrategy:
    """
    Compression strategy with both spatial and temporal denoising.
    """

    def __init__(self,
                 dataset: ArrayLike | None,
                 block_sizes: tuple[int, int] = (32, 32),
                 frame_range: int | None = None,
                 max_components: int = 20,
                 sim_conf: int = 5,
                 frame_batch_size: int = 10000,
                 max_consecutive_failures: int = 1,
                 spatial_avg_factor: int = 1,
                 temporal_avg_factor: int = 1,
                 compute_normalizer: Optional[bool] = True,
                 pixel_weighting: Optional[np.ndarray] = None,
                 device: Literal["auto", "cpu", "cuda"] = "auto",
                 # Spatial denoiser parameters
                 spatial_denoiser_epochs: int = 5,
                 spatial_denoiser_batch_size: int = 32,
                 spatial_denoiser_lr: float = 1e-4,
                 spatial_noise_variance_quantile: float = 0.7,
                 spatial_denoiser_padding: int = 12,
                 patch_h: int = 40,
                 patch_w: int = 40,
                 # Temporal denoiser parameters
                 temporal_denoiser_epochs: int = 5,
                 temporal_denoiser_batch_size: int = 128,
                 temporal_denoiser_lr: float = 1e-4,
                 temporal_noise_variance_quantile: float = 0.7,
                 ):

        self._dataset = dataset

        # PMD parameters
        self._block_sizes = block_sizes
        self._frame_range = frame_range
        self._max_components = max_components
        self._frame_batch_size = frame_batch_size
        self._spatial_avg_factor = spatial_avg_factor
        self._temporal_avg_factor = temporal_avg_factor
        self._device = device
        self._sim_conf = sim_conf
        self._max_consecutive_failures = max_consecutive_failures
        self._compute_normalizer = compute_normalizer
        self._pixel_weighting = pixel_weighting

        # Spatial denoiser parameters
        self._spatial_denoiser_epochs = spatial_denoiser_epochs
        self._spatial_denoiser_batch_size = spatial_denoiser_batch_size
        self._spatial_denoiser_lr = spatial_denoiser_lr
        self._spatial_noise_variance_quantile = spatial_noise_variance_quantile
        self._spatial_denoiser_padding = spatial_denoiser_padding
        self._patch_h = patch_h
        self._patch_w = patch_w

        # Temporal denoiser parameters
        self._temporal_denoiser_epochs = temporal_denoiser_epochs
        self._temporal_denoiser_batch_size = temporal_denoiser_batch_size
        self._temporal_denoiser_lr = temporal_denoiser_lr
        self._temporal_noise_variance_quantile = temporal_noise_variance_quantile

        self._results = None

    @property
    def dataset(self) -> ArrayLike | None:
        return self._dataset

    @dataset.setter
    def dataset(self, new_dataset: ArrayLike):
        self._dataset = new_dataset

    @property
    def block_sizes(self) -> tuple[int, int]:
        return self._block_sizes

    @block_sizes.setter
    def block_sizes(self, new_sizes: tuple[int, int]):
        self._block_sizes = new_sizes

    @property
    def frame_range(self) -> int | None:
        return self._frame_range

    @frame_range.setter
    def frame_range(self, new_frame_range: int):
        self._frame_range = new_frame_range

    @property
    def max_components(self):
        return self._max_components

    @max_components.setter
    def max_components(self, num_comps: int):
        self._max_components = num_comps

    @property
    def frame_batch_size(self) -> int:
        return self._frame_batch_size

    @frame_batch_size.setter
    def frame_batch_size(self, new_batch_size: int):
        self._frame_batch_size = new_batch_size

    @property
    def spatial_avg_factor(self) -> int:
        return self._spatial_avg_factor

    @spatial_avg_factor.setter
    def spatial_avg_factor(self, new_spatial_avg_factor: int):
        self._spatial_avg_factor = new_spatial_avg_factor

    @property
    def temporal_avg_factor(self) -> int:
        return self._temporal_avg_factor

    @temporal_avg_factor.setter
    def temporal_avg_factor(self, new_temporal_avg_factor: int):
        self._temporal_avg_factor = new_temporal_avg_factor

    @property
    def device(self) -> str:
        return self._device

    @device.setter
    def device(self, new_device: str):
        self._device = new_device

    @property
    def results(self) -> PMDArray | None:
        return self._results

    def compress(self) -> PMDArray:
        """
        Run compression with both spatial and temporal denoising:
        1. First PMD decomposition without any denoiser
        2. Train spatial denoiser on spatial components (U)
        3. Train temporal denoiser on temporal components (V)
        4. Final PMD decomposition with both denoisers
        """
        if self.dataset is None:
            raise ValueError("No dataset provided")

        print("\n" + "="*60)
        print("STEP 1: Initial PMD decomposition (no denoisers)")
        print("="*60)

        # First pass: PMD without any denoiser
        pmd_no_denoiser = pmd_decomposition(
            self.dataset,
            self._block_sizes,
            frame_range=self._frame_range,
            max_components=self._max_components,
            sim_conf=self._sim_conf,
            frame_batch_size=self._frame_batch_size,
            max_consecutive_failures=self._max_consecutive_failures,
            spatial_avg_factor=self._spatial_avg_factor,
            temporal_avg_factor=self._temporal_avg_factor,
            compute_normalizer=self._compute_normalizer,
            pixel_weighting=self._pixel_weighting,
            device=self._device
        )

        print(f"Initial PMD rank: {pmd_no_denoiser.pmd_rank}")

        # Train spatial denoiser
        print("\n" + "="*60)
        print("STEP 2: Training spatial denoiser")
        print("="*60)

        u_dense = pmd_no_denoiser.u.to_dense()
        if hasattr(self.dataset, 'shape'):
            H, W = self.dataset.shape[1], self.dataset.shape[2]
        else:
            first_frame = self.dataset[0]
            H, W = first_frame.shape

        u_reshaped = u_dense.reshape(H, W, -1)
        spatial_components = u_reshaped.permute(2, 0, 1)

        spatial_config = {
            'max_epochs': self._spatial_denoiser_epochs,
            'batch_size': self._spatial_denoiser_batch_size,
            'learning_rate': self._spatial_denoiser_lr,
            'patch_h': self._patch_h,
            'patch_w': self._patch_w,
            'device': self._device
        }

        trained_spatial_model, _ = train_spatial_denoiser(
            spatial_components,
            config=spatial_config,
            output_dir=None
        )

        spatial_denoiser = create_pmd_denoiser(
            trained_model=trained_spatial_model,
            noise_variance_quantile=self._spatial_noise_variance_quantile,
            padding=self._spatial_denoiser_padding,
            device=self._device
        )

        # Train temporal denoiser
        print("\n" + "="*60)
        print("STEP 3: Training temporal denoiser")
        print("="*60)

        v = pmd_no_denoiser.v.cpu()
        trained_temporal_model, _ = masknmf.compression.denoising.train_total_variance_denoiser(
            v,
            max_epochs=self._temporal_denoiser_epochs,
            batch_size=self._temporal_denoiser_batch_size,
            learning_rate=self._temporal_denoiser_lr
        )

        temporal_denoiser = masknmf.compression.PMDTemporalDenoiser(
            trained_temporal_model,
            self._temporal_noise_variance_quantile
        )

        # Final pass with both denoisers
        print("\n" + "="*60)
        print("STEP 4: Final PMD decomposition (with both denoisers)")
        print("="*60)

        self._results = pmd_decomposition(
            self.dataset,
            self._block_sizes,
            frame_range=self._frame_range,
            max_components=self._max_components,
            sim_conf=self._sim_conf,
            frame_batch_size=self._frame_batch_size,
            max_consecutive_failures=self._max_consecutive_failures,
            spatial_avg_factor=self._spatial_avg_factor,
            temporal_avg_factor=self._temporal_avg_factor,
            compute_normalizer=self._compute_normalizer,
            pixel_weighting=self._pixel_weighting,
            device=self._device,
            spatial_denoiser=spatial_denoiser,
            temporal_denoiser=temporal_denoiser
        )

        print(f"\nFinal PMD rank (with both denoisers): {self._results.pmd_rank}")
        print(f"Rank change: {pmd_no_denoiser.pmd_rank} -> {self._results.pmd_rank}")
        print("="*60 + "\n")

        return self._results