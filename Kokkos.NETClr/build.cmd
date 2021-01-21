
cl /c @D:\TFS_Sources\Github\Compilation\trmcnealy\Kokkos.NET\Kokkos.NETClr\build_clr.rsp D:\TFS_Sources\Github\Compilation\trmcnealy\Kokkos.NET\Kokkos.NETClr\Kokkos\ClrLibrary.cpp

link /nologo /MACHINE:X64 /CLRIMAGETYPE:IJW /CLRTHREADATTRIBUTE:STA /CLRLOADEROPTIMIZATION:SD /GUARD:NO /LTCG /CLRUNMANAGEDCODECHECK:NO ClrLibrary.obj Native.obj /ASSEMBLYMODULE:ClrLibrary.obj /out:Kokkos.NETClr.dll /LIBPATH:"C:\Program Files\dotnet\packs\Microsoft.NETCore.App.Host.win-x64\3.1.9\runtimes\win-x64\native"