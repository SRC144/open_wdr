import os
import sys
import subprocess
import shutil
from pathlib import Path
from setuptools import setup, Extension, find_packages
from setuptools.command.build_ext import build_ext
import pybind11

class CMakeExtension(Extension):
    def __init__(self, name, sourcedir=""):
        Extension.__init__(self, name, sources=[])
        self.sourcedir = os.path.abspath(sourcedir)


class CMakeBuild(build_ext):
    def build_extensions(self):
        try:
            out = subprocess.check_output(["cmake", "--version"])
        except OSError:
            raise RuntimeError(
                "CMake must be installed to build the following extensions: "
                + ", ".join(e.name for e in self.extensions)
            )

        for ext in self.extensions:
            self.build_extension(ext)

    def build_extension(self, ext):
        extdir = os.path.abspath(os.path.dirname(self.get_ext_fullpath(ext.name)))
        cmake_args = [
            "-DCMAKE_LIBRARY_OUTPUT_DIRECTORY=" + extdir,
            "-DPYTHON_EXECUTABLE=" + sys.executable,
            "-Dpybind11_DIR=" + pybind11.get_cmake_dir(),
            "-DWDR_BUILD_TESTING=OFF",
        ]

        cfg = "Release"
        build_args = ["--config", cfg]

        cmake_args += ["-DCMAKE_BUILD_TYPE=" + cfg]
        build_args += ["--parallel", "2"]

        env = os.environ.copy()
        env["CXXFLAGS"] = '{} -DVERSION_INFO=\\"{}\\"'.format(
            env.get("CXXFLAGS", ""), self.distribution.get_version()
        )
        if not os.path.exists(self.build_temp):
            os.makedirs(self.build_temp)
        subprocess.check_call(
            ["cmake", ext.sourcedir] + cmake_args, cwd=self.build_temp, env=env
        )
        subprocess.check_call(
            ["cmake", "--build", "."] + build_args, cwd=self.build_temp
        )
        if os.name == "nt":
            # The name of the generated file (e.g., coder.cp313-win_amd64.pyd)
            filename = os.path.basename(self.get_ext_fullpath(ext.name))
            
            # The path where MSVC actually put it
            src_artifact = os.path.join(extdir, cfg, filename)
            
            # The path where Python expects it
            dst_artifact = os.path.join(extdir, filename)

            if os.path.exists(src_artifact):
                print(f"Windows Fix: Moving {src_artifact} -> {dst_artifact}")
                shutil.copyfile(src_artifact, dst_artifact)


setup(
    packages=find_packages(),
    ext_modules=[CMakeExtension("wdr.coder")],
    cmdclass=dict(build_ext=CMakeBuild),
    zip_safe=False,
)

