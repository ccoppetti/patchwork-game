from dataclasses import dataclass, field
import numpy as np
from typing import Tuple, Optional
from copy import deepcopy
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class PatchShape:
    """Immutable representation of a patch's shape and rotation."""

    base_shape: np.ndarray
    rotation: int = 0

    def __post_init__(self):
        """Validate the shape array and rotation."""
        if not isinstance(self.base_shape, np.ndarray):
            raise ValueError("Shape must be a numpy array")
        if self.base_shape.dtype != np.int8:
            # Create a properly typed view of the array
            object.__setattr__(self, "base_shape", self.base_shape.astype(np.int8))
        if not np.all(np.logical_or(self.base_shape == 0, self.base_shape == 1)):
            raise ValueError("Shape array must contain only 0s and 1s")
        if self.rotation not in {0, 1, 2, 3}:
            raise ValueError("Rotation must be 0, 1, 2, or 3")

    def get_rotated(self) -> np.ndarray:
        """Returns the shape array with the current rotation applied."""
        return np.rot90(
            self.base_shape, -self.rotation
        )  # Negative for clockwise rotation

    def with_rotation(self, rotation: int) -> "PatchShape":
        """Returns a new PatchShape with the specified rotation."""
        return PatchShape(self.base_shape, rotation % 4)


@dataclass(frozen=True)
class Patch:
    """Represents a single patch piece in the game."""

    shape: PatchShape
    buttons: int
    time: int
    button_income: int
    id: int
    color: Tuple[int, int, int] = field(default=(128, 128, 128))

    def __post_init__(self):
        """Validate the patch attributes."""
        if not isinstance(self.shape, PatchShape):
            raise ValueError(f"Expected PatchShape, got {type(self.shape)}")

        if self.buttons < 0:
            raise ValueError(f"Button cost cannot be negative: {self.buttons}")

        if self.time < 0:
            raise ValueError(f"Time cost cannot be negative: {self.time}")

        if self.button_income < 0:
            raise ValueError(f"Button income cannot be negative: {self.button_income}")

        if self.id < 1:
            raise ValueError(f"Patch ID must be positive: {self.id}")

        if not isinstance(self.color, tuple) or len(self.color) != 3:
            raise ValueError(f"Invalid color format: {self.color}")

        if not all(isinstance(c, int) and 0 <= c <= 255 for c in self.color):
            raise ValueError(f"Invalid color values: {self.color}")

    @property
    def rotation(self) -> int:
        """Current rotation of the patch (0, 1, 2, or 3 for 90° increments)."""
        return self.shape.rotation

    def get_rotated_shape(self) -> np.ndarray:
        """Returns the shape array with current rotation applied."""
        return self.shape.get_rotated()

    def rotate(self, times: int = 1) -> "Patch":
        """Returns a new Patch rotated clockwise by 90° * times.

        Args:
            times: Number of 90-degree clockwise rotations to apply

        Returns:
            A new Patch instance with the rotation applied
        """
        new_shape = self.shape.with_rotation((self.rotation + times) % 4)
        return Patch(
            shape=new_shape,
            buttons=self.buttons,
            time=self.time,
            button_income=self.button_income,
            id=self.id,
            color=self.color,
        )

    def copy(self) -> "Patch":
        """Returns a deep copy of the patch."""
        return Patch(
            shape=PatchShape(self.shape.base_shape.copy(), self.shape.rotation),
            buttons=self.buttons,
            time=self.time,
            button_income=self.button_income,
            id=self.id,
            color=self.color,
        )

    @classmethod
    def from_array(
        cls,
        shape_array: np.ndarray,
        buttons: int,
        time: int,
        button_income: int,
        id: int,
        color: Optional[Tuple[int, int, int]] = None,
    ) -> "Patch":
        """Factory method to create a Patch from a shape array.

        Args:
            shape_array: 2D numpy array of 0s and 1s defining the patch shape
            buttons: Button cost of the patch
            time: Time cost of the patch
            button_income: Buttons generated by the patch
            id: Unique identifier for the patch
            color: RGB color tuple (defaults to gray)

        Returns:
            A new Patch instance
        """
        try:
            shape = PatchShape(shape_array)
            return cls(
                shape=shape,
                buttons=buttons,
                time=time,
                button_income=button_income,
                id=id,
                color=color or (128, 128, 128),
            )
        except ValueError as e:
            logger.error(f"Failed to create patch with ID {id}: {str(e)}")
            raise
