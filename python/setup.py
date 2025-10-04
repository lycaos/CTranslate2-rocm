import glob
import os
import shutil
import sys

import pybind11

from pybind11.setup_helpers import ParallelCompile
from setuptools import Extension, find_packages, setup
from setuptools.command.build_ext import build_ext

base_dir = os.path.dirname(os.path.abspath(__file__))
include_dirs = [pybind11.get_include()]
library_dirs = []


def _get_long_description():
    readme_path = os.path.join(base_dir, "README.md")
    if not os.path.exists(readme_path):
        return ""
    with open(readme_path, encoding="utf-8") as readme_file:
        return readme_file.read()


def _get_project_version():
    version_path = os.path.join(base_dir, "ctranslate2", "version.py")
    version = {}
    with open(version_path, encoding="utf-8") as fp:
        exec(fp.read(), version)
    return version["__version__"]


def _maybe_add_library_root(lib_name):
    if "%s_ROOT" % lib_name in os.environ:
        root = os.environ["%s_ROOT" % lib_name]
        include_dirs.append("%s/include" % root)
        for lib_dir in ("lib", "lib64"):
            path = "%s/%s" % (root, lib_dir)
            if os.path.exists(path):
                library_dirs.append(path)
                break


_maybe_add_library_root("CTRANSLATE2")

cflags = ["-std=c++17", "-fvisibility=hidden"]
ldflags = []
package_data = {}

# Custom build_ext to copy libctranslate2.so into the package
class CustomBuildExt(build_ext):
    def run(self):
        super().run()
        # Copy libctranslate2.so* into the ctranslate2 package directory
        if sys.platform.startswith("linux"):
            # Try CTRANSLATE2_ROOT first, then CONDA_PREFIX, then system paths
            ct2_root = os.environ.get("CTRANSLATE2_ROOT") or os.environ.get("CONDA_PREFIX")
            
            if ct2_root:
                lib_src_dir = None
                for lib_dir in ("lib", "lib64"):
                    path = os.path.join(ct2_root, lib_dir)
                    if os.path.exists(path):
                        lib_src_dir = path
                        break
                
                if lib_src_dir:
                    # Find all libctranslate2.so* files
                    lib_pattern = os.path.join(lib_src_dir, "libctranslate2.so*")
                    lib_files = glob.glob(lib_pattern)
                    
                    if lib_files:
                        # Copy to package directory in build_lib
                        package_dir = os.path.join(self.build_lib, "ctranslate2")
                        os.makedirs(package_dir, exist_ok=True)
                        
                        print(f"\n{'='*60}")
                        print("Copying shared libraries to wheel:")
                        print(f"{'='*60}")
                        for lib_file in lib_files:
                            # Skip symlinks, we'll recreate them
                            if os.path.islink(lib_file):
                                continue
                            
                            dest = os.path.join(package_dir, os.path.basename(lib_file))
                            print(f"  {lib_file} -> {dest}")
                            shutil.copy2(lib_file, dest)
                        
                        # Create symlinks for versioned libraries
                        for lib_file in lib_files:
                            basename = os.path.basename(lib_file)
                            if os.path.islink(lib_file):
                                target = os.readlink(lib_file)
                                link_path = os.path.join(package_dir, basename)
                                if not os.path.exists(link_path):
                                    os.symlink(target, link_path)
                                    print(f"  Created symlink: {basename} -> {target}")
                        print(f"{'='*60}\n")
                    else:
                        print(f"WARNING: No libctranslate2.so* found in {lib_src_dir}")
                else:
                    print(f"WARNING: No lib directory found in {ct2_root}")
            else:
                print("WARNING: CTRANSLATE2_ROOT or CONDA_PREFIX not set, skipping library copy")

if sys.platform == "darwin":
    # std::visit requires macOS 10.14
    cflags.append("-mmacosx-version-min=10.14")
    ldflags.append("-Wl,-rpath,/usr/local/lib")
elif sys.platform == "win32":
    cflags = ["/std:c++17", "/d2FH4-"]
    package_data["ctranslate2"] = ["*.dll"]
elif sys.platform.startswith("linux"):
    # Include shared libraries in the wheel
    package_data["ctranslate2"] = ["*.so*"]
    # Set RPATH to look in the same directory as _ext.so
    ldflags.append("-Wl,-rpath,$ORIGIN")

ctranslate2_module = Extension(
    "ctranslate2._ext",
    sources=glob.glob(os.path.join("cpp", "*.cc")),
    extra_compile_args=cflags,
    extra_link_args=ldflags,
    include_dirs=include_dirs,
    library_dirs=library_dirs,
    libraries=["ctranslate2"],
)

ParallelCompile("CMAKE_BUILD_PARALLEL_LEVEL").install()

setup(
    name="ctranslate2",
    version=_get_project_version(),
    license="MIT",
    description="Fast inference engine for Transformer models",
    long_description=_get_long_description(),
    long_description_content_type="text/markdown",
    author="OpenNMT",
    url="https://opennmt.net",
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Environment :: GPU :: NVIDIA CUDA :: 12 :: 12.0",
        "Environment :: GPU :: NVIDIA CUDA :: 12 :: 12.1",
        "Environment :: GPU :: NVIDIA CUDA :: 12 :: 12.2",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3 :: Only",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Programming Language :: Python :: 3.13",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    project_urls={
        "Documentation": "https://opennmt.net/CTranslate2",
        "Forum": "https://forum.opennmt.net",
        "Gitter": "https://gitter.im/OpenNMT/CTranslate2",
        "Source": "https://github.com/OpenNMT/CTranslate2",
    },
    keywords="opennmt nmt neural machine translation cuda mkl inference quantization",
    packages=find_packages(exclude=["bin"]),
    package_data=package_data,
    ext_modules=[ctranslate2_module],
    cmdclass={"build_ext": CustomBuildExt},
    python_requires=">=3.9",
    install_requires=[
        "setuptools",
        "numpy",
        "pyyaml>=5.3,<7",
    ],
    entry_points={
        "console_scripts": [
            "ct2-fairseq-converter=ctranslate2.converters.fairseq:main",
            "ct2-marian-converter=ctranslate2.converters.marian:main",
            "ct2-openai-gpt2-converter=ctranslate2.converters.openai_gpt2:main",
            "ct2-opennmt-py-converter=ctranslate2.converters.opennmt_py:main",
            "ct2-opennmt-tf-converter=ctranslate2.converters.opennmt_tf:main",
            "ct2-opus-mt-converter=ctranslate2.converters.opus_mt:main",
            "ct2-transformers-converter=ctranslate2.converters.transformers:main",
        ],
    },
)
