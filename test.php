<?php

// FFI
if (!extension_loaded('ffi')) {
    die('FFI extension is not loaded.');
}

// FFI
$ffi = FFI::cdef("
    typedef struct {
        int major;
        int minor;
        int subminor;
    } vDSP_Version;

    void vDSP_initialize(vDSP_Version *version);
", "/System/Library/Frameworks/Accelerate.framework/Versions/Current/Frameworks/vecLib.framework/Versions/Current/libvDSP.dylib");

// vDSP_Version
$version = $ffi->new('vDSP_Version');

// vDSP_initialize
$ffi->vDSP_initialize(FFI::addr($version));

// 
echo "vecLib Version: {$version->major}.{$version->minor}.{$version->subminor}\n";

