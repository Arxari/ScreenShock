### ScreenShock
Shocks you based on what's on your screen
Currently tested and working on Hyprland

I'll ensure support for DEs and some GUI or TUI, just gimme a bit

How to use:
- Git clone the repo
- Create a ```reference_images``` folder with the images you want to be used as reference for shocks, for example these can be UI elements from a game that indicates you died
- Run the script
-- -
You need these packages:
- For Mint, Ubuntu and the like
```
sudo apt install grim tesseract-ocr libopencv-dev python3 python3-pip build-essential
```
- For Arch
sudo pacman -S grim tesseract-ocr libopencv-dev python python-pip build-essential
