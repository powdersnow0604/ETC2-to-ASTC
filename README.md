# ETC2-to-ASTC

## Overview
This project provides a tool for converting ETC2 textures to ASTC format. It is designed to work with libraries built from the following repositories:

- [ARM-software/astc-encoder](https://github.com/ARM-software/astc-encoder/)
- [wolfpld/etcpak](https://github.com/wolfpld/etcpak)

## Requirements
- You must build the libraries from the above repositories and link them to this project.
- One of the headers from etcpak, `BlockData.hpp`, should be replaced with the version provided in this repository. When building the library in the original etcpak repository, you can use the original header.

## Usage
- This project does **not** use command-line arguments. Instead, you should modify `main.cpp` directly to set input/output files or parameters as needed.
- Make sure to update the linking in `main.cpp` to use the correct libraries and headers as described above.

## Notes
- For more information on building the required libraries, refer to the documentation in their respective repositories. 