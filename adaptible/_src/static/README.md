# Cropper Tool

```shell
brew install imagemagick
```

### To crop an image

```shell
cd adaptible/_src/static
sh ./crop.sh -i gemini.png -o gemini_cropped.png -p 60
```

### To crop an image AND remove its background

(based on its edge colors)

```shell
cd adaptible/_src/static
sh ./crop.sh -i gemini.png -o gemini_cropped_transparent_bg.png -p 60 -r
```

