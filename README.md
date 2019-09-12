Image Style Transfer using CNNs
---

Experimenting with a Pytorch implementation of ['A Neural Algorithm of Artistic Style' by L. Gatys, A. Ecker, and M. Bethge](http://arxiv.org/abs/1508.06576).


Dependencies
--
* Pytorch (version >= 0.4.0)

Usage
--

* Test example script

```
python main.py --cuda-device-no 0 --target-content-filename sample_images/content_images/chicago.jpg --target-style-filename sample_images/style_images/mondrian.jpg --save-filename stylized.png
```

Example images
--

![](https://github.com/HespK/style-transfer-pytorch/blob/master/sample_images/example/example.jpg)
