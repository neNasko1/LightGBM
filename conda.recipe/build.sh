sh ./build-python.sh bdist_wheel --nomp --no-isolation
python3 -m pip install dist/*.whl --prefix=${PREFIX}
