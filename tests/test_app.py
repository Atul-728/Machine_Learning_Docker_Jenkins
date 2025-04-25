import unittest
import sys
import os

# Add parent directory to path to import app
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app import app

class FlaskAppTests(unittest.TestCase):
    
    def setUp(self):
        self.app = app.test_client()
        self.app.testing = True
    
    def test_home_status_code(self):
        result = self.app.get('/')
        self.assertEqual(result.status_code, 200)
    
    def test_about_status_code(self):
        result = self.app.get('/about')
        self.assertEqual(result.status_code, 200)
    
    def test_predictor_status_code(self):
        result = self.app.get('/predictor')
        self.assertEqual(result.status_code, 200)

if __name__ == '__main__':
    unittest.main()
