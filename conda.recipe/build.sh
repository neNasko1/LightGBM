create_isolated_source_dir() {
    rm -rf \
        ./lightgbm-python \
        ./lightgbm \
        ./python-package/build \
        ./python-package/build_cpp \
        ./python-package/compile \
        ./python-package/dist \
        ./python-package/lightgbm.egg-info

    cp -R ./python-package ./lightgbm-python

    cp LICENSE ./lightgbm-python/
    cp VERSION.txt ./lightgbm-python/lightgbm/VERSION.txt

    cp -R ./cmake ./lightgbm-python
    cp CMakeLists.txt ./lightgbm-python
    cp -R ./include ./lightgbm-python
    cp -R ./src ./lightgbm-python
    cp -R ./swig ./lightgbm-python
    cp -R ./windows ./lightgbm-python

    # include only specific files from external_libs, to keep the package
    # small and avoid redistributing code with licenses incompatible with
    # LightGBM's license

    ######################
    # fast_double_parser #
    ######################
    mkdir -p ./lightgbm-python/external_libs/fast_double_parser
    cp \
        external_libs/fast_double_parser/CMakeLists.txt \
        ./lightgbm-python/external_libs/fast_double_parser/CMakeLists.txt
    cp \
        external_libs/fast_double_parser/LICENSE* \
        ./lightgbm-python/external_libs/fast_double_parser/

    mkdir -p ./lightgbm-python/external_libs/fast_double_parser/include/
    cp \
        external_libs/fast_double_parser/include/fast_double_parser.h \
        ./lightgbm-python/external_libs/fast_double_parser/include/

    #######
    # fmt #
    #######
    mkdir -p ./lightgbm-python/external_libs/fmt
    cp \
        external_libs/fast_double_parser/CMakeLists.txt \
        ./lightgbm-python/external_libs/fmt/CMakeLists.txt
    cp \
        external_libs/fmt/LICENSE* \
        ./lightgbm-python/external_libs/fmt/

    mkdir -p ./lightgbm-python/external_libs/fmt/include/fmt
    cp \
        external_libs/fmt/include/fmt/*.h \
        ./lightgbm-python/external_libs/fmt/include/fmt/

    #########
    # Eigen #
    #########
    mkdir -p ./lightgbm-python/external_libs/eigen/Eigen
    cp \
        external_libs/eigen/CMakeLists.txt \
        ./lightgbm-python/external_libs/eigen/CMakeLists.txt

    modules="Cholesky Core Dense Eigenvalues Geometry Householder Jacobi LU QR SVD"
    for eigen_module in ${modules}; do
        cp \
            external_libs/eigen/Eigen/${eigen_module} \
            ./lightgbm-python/external_libs/eigen/Eigen/${eigen_module}
        if [ ${eigen_module} != "Dense" ]; then
            mkdir -p ./lightgbm-python/external_libs/eigen/Eigen/src/${eigen_module}/
            cp \
                -R \
                external_libs/eigen/Eigen/src/${eigen_module}/* \
                ./lightgbm-python/external_libs/eigen/Eigen/src/${eigen_module}/
        fi
    done

    mkdir -p ./lightgbm-python/external_libs/eigen/Eigen/misc
    cp \
        -R \
        external_libs/eigen/Eigen/src/misc \
        ./lightgbm-python/external_libs/eigen/Eigen/src/misc/

    mkdir -p ./lightgbm-python/external_libs/eigen/Eigen/plugins
    cp \
        -R \
        external_libs/eigen/Eigen/src/plugins \
        ./lightgbm-python/external_libs/eigen/Eigen/src/plugins/

    ###################
    # compute (Boost) #
    ###################
    mkdir -p ./lightgbm-python/external_libs/compute
    cp \
        -R \
        external_libs/compute/include \
        ./lightgbm-python/external_libs/compute/include/
}

create_isolated_source_dir

python -m build --wheel --outdir ../dist --config-setting=cmake.define.USE_OPENMP=OFF lightgbm-python --no-isolation
