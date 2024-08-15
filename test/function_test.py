import unittest

from app.function import add


class FunctionTestCase(unittest.TestCase):
    def test_add(self):
        self.assertEqual(add(1, 1), 2)
        self.assertEqual(add("2", 2), 4)
        self.assertEqual(add("3", "3"), 6)

        self.assertNotEqual(add(1, 1), 3)
        self.assertNotEqual(add("2", 2), 5)
        self.assertNotEqual(add("3", "3"), 7)


if __name__ == "__main__":
    unittest.main()
