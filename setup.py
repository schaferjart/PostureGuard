"""
Build PostureGuard as a macOS .app bundle using py2app.
Run: python3 setup.py py2app
"""
from setuptools import setup

APP = ['postureguard.py']
OPTIONS = {
    'argv_emulation': False,
    'plist': {
        'CFBundleName': 'PostureGuard',
        'CFBundleDisplayName': 'PostureGuard',
        'CFBundleIdentifier': 'com.postureguard.app',
        'CFBundleVersion': '1.1.0',
        'CFBundleShortVersionString': '1.1.0',
        'LSUIElement': True,  # No dock icon!
        'NSCameraUsageDescription': 'PostureGuard needs camera access to monitor your posture.',
    },
    'packages': ['cv2', 'mediapipe', 'numpy', 'rumps', 'google'],
    'includes': ['mediapipe.python.solutions.pose', 'mediapipe.python.solutions.face_mesh',
                  'posture_core'],
}

DATA_FILES = ['camera_preview.py', 'posture_core.py']

setup(
    app=APP,
    name='PostureGuard',
    data_files=DATA_FILES,
    options={'py2app': OPTIONS},
    setup_requires=['py2app'],
)
