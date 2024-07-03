import os
import subprocess

from setuptools import find_packages, setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension


def make_cuda_ext(name, module, sources):
    cuda_ext = CUDAExtension(
        name='%s.%s' % (module, name),
        sources=[os.path.join(*module.split('.'), src) for src in sources]
    )
    return cuda_ext


if __name__ == '__main__':
            # 'numba',
            # 'SharedArray',
            # 'pycocotools',
            # 'terminaltables',
            # 'einops',
            # 'llvmlite',
            # 'timm'
    setup(
        name='object_detector',
        install_requires=[
            'numpy',
            'easydict',
            'pyyaml',
            'tqdm',
            'tensorboardX',
        ],

        packages=find_packages(exclude=['tools', 'data', 'output']),
        cmdclass={
            'build_ext': BuildExtension,
        },
        ext_modules=[
            make_cuda_ext(
                name='iou3d_nms_cuda',
                module='graphvoi.ops.iou3d_nms',
                sources=[
                    'src/iou3d_cpu.cpp',
                    'src/iou3d_nms_api.cpp',
                    'src/iou3d_nms.cpp',
                    'src/iou3d_nms_kernel.cu',
                ]
            ),
            make_cuda_ext(
                name='roiaware_pool3d_cuda',
                module='graphvoi.ops.roiaware_pool3d',
                sources=[
                    'src/roiaware_pool3d.cpp',
                    'src/roiaware_pool3d_kernel.cu',
                ]
            ),
            # make_cuda_ext(
            #     name='roipoint_pool3d_cuda',
            #     module='graphvoi.ops.roipoint_pool3d',
            #     sources=[
            #         'src/roipoint_pool3d.cpp',
            #         'src/roipoint_pool3d_kernel.cu',
            #     ]
            # ),
            make_cuda_ext(
                name='pointnet2_stack_cuda',
                module='graphvoi.ops.pointnet2.pointnet2_stack',
                sources=[
                    'src/pointnet2_api.cpp',
                    'src/ball_query.cpp',
                    'src/ball_query_gpu.cu',
                    'src/group_points.cpp',
                    'src/group_points_gpu.cu',
                    'src/sampling.cpp',
                    'src/sampling_gpu.cu', 
                    'src/interpolate.cpp', 
                    'src/interpolate_gpu.cu',
                    'src/voxel_query.cpp', 
                    'src/voxel_query_gpu.cu',
                ],
            ),
            make_cuda_ext(
                name='pointnet2_batch_cuda',
                module='graphvoi.ops.pointnet2.pointnet2_batch',
                sources=[
                    'src/pointnet2_api.cpp',
                    'src/ball_query.cpp',
                    'src/ball_query_gpu.cu',
                    'src/group_points.cpp',
                    'src/group_points_gpu.cu',
                    'src/interpolate.cpp',
                    'src/interpolate_gpu.cu',
                    'src/sampling.cpp',
                    'src/sampling_gpu.cu',

                ],
            ),
            make_cuda_ext(
                name='patch_ops_cuda',
                module='graphvoi.ops.patch_ops',
                sources=[
                    'src/patch_ops_api.cpp',
                    'src/patch_query.cpp',
                    'src/patch_query_gpu.cu',
                    'src/roipatch_dfvs_pool3d.cpp',
                    'src/roipatch_dfvs_pool3d_gpu.cu',
                ],
            ),
            # make_cuda_ext(
            #     name='sst_ops_cuda',
            #     module='graphvoi.ops.sst_ops',
            #     sources=[
            #         'src/sst_ops_api.cpp',
            #         'src/sst_ops.cpp',
            #         'src/sst_ops_gpu.cu'
            #     ]
            # ),
        ],
    )
