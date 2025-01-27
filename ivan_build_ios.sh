mkdir build
cd build
cmake -G Xcode .. \
            -DGGML_METAL_USE_BF16=ON \
            -DGGML_METAL_EMBED_LIBRARY=ON \
            -DLLAMA_BUILD_EXAMPLES=OFF \
            -DLLAMA_BUILD_TESTS=OFF \
            -DLLAMA_BUILD_SERVER=OFF \
            -DCMAKE_SYSTEM_NAME=iOS \
            -DCMAKE_OSX_DEPLOYMENT_TARGET=14.0 \
            -DCMAKE_XCODE_ATTRIBUTE_DEVELOPMENT_TEAM=ggml \
            -DLLAMA_BUILD_COMMON=ON \
	    -DBUILD_SHARED_LIBS=ON \
	    -DCMAKE_FRAMEWORK=ON
cmake --build . --config Release -j $(sysctl -n hw.logicalcpu) -- CODE_SIGNING_ALLOWED=NO
sudo cmake --install . --config Release

cd ..

xcodebuild -scheme llama-Package -destination 'generic/platform=iOS'

