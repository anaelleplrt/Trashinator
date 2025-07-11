import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"  # ← Optionnel si le crash persiste
import matplotlib
matplotlib.use('Agg')  # ← Important ! Fixe le backend
import sqlite3
from flask import Flask, request, redirect, render_template, url_for, jsonify, send_file
import cv2
import json
from werkzeug.utils import secure_filename
from PIL import Image as PILImage
import numpy as np
from collections import defaultdict, Counter
from datetime import datetime
import sqlite3
from PIL.ExifTags import TAGS, GPSTAGS
from flask import send_file
import matplotlib.pyplot as plt
import io
import math
from ultralytics import YOLO
import random
from random import choice, uniform
from geopy.geocoders import Nominatim
import time