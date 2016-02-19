#!/bin/bash
# This script starts the image recognition API

cd /opt/face_recognizer/www/cv_api

python manage.py migrate

python manage.py runserver 0.0.0.0:${API_PORT}

