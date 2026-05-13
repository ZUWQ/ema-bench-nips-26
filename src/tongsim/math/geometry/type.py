"""
tongsim.math.geometry.type
"""

# ruff: noqa: N812
from pyglm import glm as _glm
from pyglm.glm import mat4 as Mat4
from pyglm.glm import mat4_cast, translate
from pyglm.glm import quat as Quaternion
from pyglm.glm import scale as glm_scale
from pyglm.glm import vec3 as Vector3

__all__ = ["AABB", "Mat4", "Pose", "Quaternion", "Transform", "Vector3"]


class Pose:
    """
    Pose 类，提供 location 和 rotation 的轻量级只读结构封装，用于位置和旋转统一传递。
    """

    __slots__ = ("location", "rotation")

    def __init__(
        self, location: Vector3 | None = None, rotation: Quaternion | None = None
    ):
        self.location = location if location is not None else Vector3(0.0, 0.0, 0.0)
        self.rotation = (
            rotation if rotation is not None else Quaternion(1.0, 0.0, 0.0, 0.0)
        )

    def __repr__(self) -> str:
        return f"Pose(location={self.location}, rotation={self.rotation})"

    def copy(self) -> "Pose":
        """
        返回该 Pose 的 deepcopy。
        """
        return Pose(Vector3(self.location), Quaternion(self.rotation))

    def __eq__(self, other: object) -> bool:
        return (
            isinstance(other, Pose)
            and self.location == other.location
            and self.rotation == other.rotation
        )

    def to_transform(self) -> "Transform":
        """
        将当前 Pose 转换为 Transform 。
        """
        return Transform(self.location, self.rotation, Vector3(1.0, 1.0, 1.0))


class Transform:
    """
    Transform 类，封装 location、rotation、scale 三个字段。

    提供统一的空间变换数据结构，与 Unreal Engine Transform 对齐。
    """

    __slots__ = ("location", "rotation", "scale")

    def __init__(
        self,
        location: Vector3 | None = None,
        rotation: Quaternion | None = None,
        scale: Vector3 | None = None,
    ):
        self.location = location if location is not None else Vector3(0.0, 0.0, 0.0)
        self.rotation = (
            rotation if rotation is not None else Quaternion(1.0, 0.0, 0.0, 0.0)
        )
        self.scale = scale if scale is not None else Vector3(1.0, 1.0, 1.0)

    def __repr__(self) -> str:
        return (
            f"Transform(location={self.location}, "
            f"rotation={self.rotation}, scale={self.scale})"
        )

    def copy(self) -> "Transform":
        """
        返回当前 Transform 的 deepcopy。
        """
        return Transform(
            Vector3(self.location),
            Quaternion(self.rotation),
            Vector3(self.scale),
        )

    def __eq__(self, other: object) -> bool:
        return (
            isinstance(other, Transform)
            and self.location == other.location
            and self.rotation == other.rotation
            and self.scale == other.scale
        )

    def to_matrix(self) -> Mat4:
        """
        返回当前 Transform 对应的 4x4 仿射变换矩阵。
        顺序为: scale → rotate → translate。
        """
        t = translate(Mat4(1.0), self.location)
        r = mat4_cast(self.rotation)
        s = glm_scale(Mat4(1.0), self.scale)
        return t * r * s  # 注意矩阵右乘顺序

    def transform_vector3(self, point: Vector3) -> Vector3:
        """
        将当前 Transform 应用于一个 3D 点，返回变换后的结果。
        """
        m = self.to_matrix()
        p = m * _glm.vec4(point, 1.0)  # 使用齐次坐标
        return Vector3(p.x, p.y, p.z)

    def inverse(self) -> "Transform":
        """
        返回当前 Transform 的逆变换。
        注意: 先逆 scale、再逆 rotate、最后逆 translate。
        """
        if self.scale.x == 0 or self.scale.y == 0 or self.scale.z == 0:
            raise ValueError(f"Cannot invert Transform with zero scale: {self.scale}")
        inv_scale = Vector3(
            1.0 / self.scale.x,
            1.0 / self.scale.y,
            1.0 / self.scale.z,
        )
        inv_rot = _glm.inverse(self.rotation)
        inv_loc = -inv_rot * (inv_scale * self.location)
        return Transform(inv_loc, inv_rot, inv_scale)


class AABB:
    """
    表示一个三维空间中的轴对齐包围盒（Axis-Aligned Bounding Box）。

    Attributes:
        min (Vector3): 包围盒的最小点（x, y, z 坐标最小）。
        max (Vector3): 包围盒的最大点（x, y, z 坐标最大）。
    """

    __slots__ = ("max", "min")

    def __init__(self, min: Vector3, max: Vector3):
        self.min = min
        self.max = max

    def __repr__(self) -> str:
        return f"AABB(min={self.min}, max={self.max})"

    def deepcopy(self) -> "AABB":
        """
        返回该 AABB 的 deepcopy。
        """
        return AABB(Vector3(self.min), Vector3(self.max))

    def center(self) -> Vector3:
        """
        返回 AABB 的中心点。
        """
        return (self.min + self.max) * 0.5

    def extent(self) -> Vector3:
        """
        返回 AABB 的尺寸（宽度、高度、深度）。
        """
        return self.max - self.min

    def contains_point(self, point: Vector3) -> bool:
        """
        判断一个点是否在该 AABB 内部。

        Args:
            point (Vector3): 要判断的点。

        Returns:
            bool: 是否在内部。
        """
        return (
            self.min.x <= point.x <= self.max.x
            and self.min.y <= point.y <= self.max.y
            and self.min.z <= point.z <= self.max.z
        )
