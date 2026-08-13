import os
import shutil

from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CppExtension

ext_modules = [
    CppExtension(
        name="vllm_kunlun._kunlun",
        sources=["vllm_kunlun/csrc/utils.cpp"],
        include_dirs=[
            "vllm_kunlun/csrc",
        ],
        extra_compile_args=["-O3"],
    )
]


class CustomBuildExt(BuildExtension):
    def run(self):
        super().run()
        for ext in self.extensions:
            ext_path = self.get_ext_fullpath(ext.name)
            file_name = os.path.basename(ext_path)
            target_path = os.path.join("vllm_kunlun", file_name)

            if os.path.exists(target_path):
                os.remove(target_path)
            shutil.copyfile(ext_path, target_path)
            print(f"[BuildExt] Copied {ext_path} -> {target_path}")


setup(
    ext_modules=ext_modules,
    cmdclass={"build_ext": CustomBuildExt},
)
