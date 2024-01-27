<?php
namespace Rindow\OpenBLAS\FFI;

use FFI;
//use FFI\Env\Runtime as FFIEnvRuntime;
//use FFI\Env\Status as FFIEnvStatus;
//use FFI\Location\Locator as FFIEnvLocator;
use FFI\Exception as FFIException;
use RuntimeException;

class OpenBLASFactory
{
    private static ?FFI $ffi = null;
    private static ?FFI $ffiLapacke = null;
    protected array $libs_win = ['libopenblas.dll'];
    protected array $libs_linux = ['libopenblas.so','libopenblas.so.0'];
    protected array $lapacke_win = ['libopenblas.dll'];
    protected array $lapacke_linux = ['liblapacke.so','liblapacke.so.3'];

    public function __construct(
        string $headerFile=null,
        array $libFiles=null,
        string $lapackeHeader=null,
        array $lapackeLibs=null,
        )
    {
        if(self::$ffi!==null) {
            return;
        }
        //
        // blas
        //
        $headerFile = $headerFile ?? __DIR__.'/openblas.h';
        if($libFiles==null) {
            if(PHP_OS=='Linux') {
                $libFiles = $this->libs_linux;
            } elseif(PHP_OS=='WINNT') {
                $libFiles = $this->libs_win;
            } else {
                throw new RuntimeException('Unknown operating system: "'.PHP_OS.'"');
            }
        }
        $code = file_get_contents($headerFile);
        // ***************************************************************
        // FFI Locator is incompletely implemented. It is often not found.
        // ***************************************************************
        //$pathname = FFIEnvLocator::resolve(...$libFiles);
        //if($pathname) {
        //    $ffi = FFI::cdef($code,$pathname);
        //    self::$ffi = $ffi;
        //}
        foreach ($libFiles as $filename) {
            $ffi = null;
            try {
                $ffi = FFI::cdef($code,$filename);
            } catch(FFIException $e) {
                continue;
            }
            self::$ffi = $ffi;
            break;
        }
        //
        // lapacke
        //
        $lapackeHeader = $lapackeHeader ?? __DIR__ . '/lapacke.h';
        if($lapackeLibs==null) {
            if(PHP_OS=='Linux') {
                $lapackeLibs = $this->lapacke_linux;
            } elseif(PHP_OS=='WINNT') {
                $lapackeLibs = $this->lapacke_win;
            } else {
                throw new RuntimeException('Unknown operating system: "'.PHP_OS.'"');
            }
        }
        $code = file_get_contents($lapackeHeader);
        // ***************************************************************
        // FFI Locator is incompletely implemented. It is often not found.
        // ***************************************************************
        //if(PHP_OS=='Linux') {
        //    $libFiles = $this->lapackeLibs;
        //    $pathname = FFIEnvLocator::resolve(...$libFiles);
        //}
        //if($pathname) {
        //    $code = file_get_contents($headerFile);
        //    $ffi = FFI::cdef($code,$pathname);
        //    self::$ffiLapacke = $ffi;
        //}
        foreach ($lapackeLibs as $filename) {
            $ffiLapacke = null;
            try {
                $ffiLapacke = FFI::cdef($code,$filename);
            } catch(FFIException $e) {
                continue;
            }
            self::$ffiLapacke = $ffiLapacke;
            break;
        }
    }

    public function isAvailable() : bool
    {
        return self::$ffi!==null;
        //$isAvailable = FFIEnvRuntime::isAvailable();
        //if(!$isAvailable) {
        //    return false;
        //}
        //$pathname = FFIEnvLocator::resolve(...$this->libs);
        //return $pathname!==null;
    }

    public function Blas() : Blas
    {
        if(self::$ffi==null) {
            throw new RuntimeException('openblas library not loaded.');
        }
        return new Blas(self::$ffi);
    }

    public function Lapack() : Lapack
    {
        if(self::$ffiLapacke==null) {
            throw new RuntimeException('lapacke library not loaded.');
        }
        return new Lapack(self::$ffiLapacke);
    }
}
