### ScreenShock
Shocks you based on what's on your screen

How to use:
- Git clone the repo
- Create a ```reference_images``` folder with the images you want to be used as reference for shocks, for example these can be UI elements from a game that indicates you died
- Install the dependencies
- Run the script
-- -
You need these packages:
- For Mint, Ubuntu and the like
just cd into the folder and run `pip install -r requirements.txt`


- For Arch
```
sudo pacman -S grim python python-requests python-opencv python-numpy
```
Or you can also do the pip method
Replace `grim` with `spectacle` if you're on KDE and `gnome-screenshot` if you're on GNOME
