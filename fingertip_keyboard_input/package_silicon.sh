#!/bin/bash

echo "Getting mediapipe path"
python_command="import mediapipe, os.path; print(os.path.dirname(mediapipe.__file__))"
mediapipe_path=$(python3.11 -c "$python_command")

echo "Writing bundled executable to ./dist"
/usr/local/bin/python3.11 -m PyInstaller --noconfirm --add-data "data/template_image_new_wide_data/:data/template_image_new_wide_data/" --add-data "mnist_model/saved_models/:mnist_model/saved_models/" --add-data "emnist_model/saved_models/:emnist_model/saved_models/" --add-data "$mediapipe_path:mediapipe" mouse_and_keyboard_control_topkg.py