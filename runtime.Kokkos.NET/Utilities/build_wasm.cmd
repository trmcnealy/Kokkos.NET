:: #!/bin/bash

@ECHO OFF
cls


set "INCLUDES=-isystem D:/AssemblyCache/Wasm/include -isystem D:\POSIX/clang64/x86_64-w64-mingw32/include"

set "DEFINES=-D NDEBUG -D _NDEBUG -D MBCS -D _MBCS -D _M_AMD64 -D x86 -D X86 -D _x86_ -D _X86_ -D __i386__ -D WIN32 -D _WIN32 -D _WINDOWS -D HAVE_WINNT_IGNORE_VOID -D WIN32_LEAN_AND_MEAN -D _GNU_SOURCE -D _USE_MATH_DEFINES -D _GLIBCXX_USE_CXX11_ABI=0 -U _MSC_VER -Wno-ignored-attributes"

set "FLAGS=-fdeclspec -fms-extensions -fforce-emit-vtables -mms-bitfields -fcoroutines-ts -fforce-enable-int128 -fcxx-exceptions -fpermissive -fno-short-enums -fshort-wchar -W#pragma-messages -Xclang -flto-visibility-public-std -fvisibility-ms-compat -fcaret-diagnostics -fdiagnostics-show-template-tree -fcolor-diagnostics -fdiagnostics-parseable-fixits -fdiagnostics-format=msvc -ftemplate-backtrace-limit=0 -ferror-limit=10 -femulated-tls -march=native -mtune=native -fvectorize -fslp-vectorize -fno-debug-macro -O3 -fomit-frame-pointer -ffp-model=precise -ffp-contract=off -fno-fast-math -fno-signed-zeros -freciprocal-math -Xarch_host -fopenmp=libomp -Xarch_host -fopenmp-simd"


D:\AssemblyCache\LLVM\bin\clang++.exe -c --sysroot="D:/AssemblyCache/Wasm/" --target=wasm32-unknown-wasi -o "D:/TFS_Sources/EngineeringTools/bin/Wasm/SharedMemory.obj" %INCLUDES% %DEFINES% %FLAGS% SharedMemory.cpp





D:\AssemblyCache\LLVM\bin\wasm-ld.exe --no-entry -LD:/AssemblyCache/Wasm/lib --no-whole-archive -lc -lc++ -lc++abi -lm -ldl -lrt -lwasi-emulated-mman -o "D:/TFS_Sources/EngineeringTools/bin/Wasm/SharedMemory.wasm" --whole-archive "D:/TFS_Sources/EngineeringTools/bin/Wasm/SharedMemory.obj"








