## Benchmark results

- model: Resnet50
- backend: tensorflow 2.4.1

### GPU

|    |     nhwc |     nchw |
|---:|---------:|---------:|
|  0 | 3.72646  | 1.33063  |
|  1 | 0.512893 | 0.583547 |
|  2 | 0.499223 | 0.568012 |
|  3 | 0.506143 | 0.568648 |
|  4 | 0.501938 | 0.56862  |
|  5 | 0.502459 | 0.566541 |
|  6 | 0.511426 | 0.573368 |
|  7 | 0.498219 | 0.56866  |
|  8 | 0.513217 | 0.623998 |
|  9 | 0.498229 | 0.570358 |
| 10 | 0.57552  | 0.568504 |
| 11 | 0.509903 | 0.56607  |
| 12 | 0.506191 | 0.5997   |
| 13 | 0.506174 | 0.562624 |
| 14 | 0.510592 | 0.56846  |
| 15 | 0.498955 | 0.568063 |
| 16 | 0.506306 | 0.571993 |
| 17 | 0.506009 | 0.641682 |
| 18 | 0.508422 | 0.572816 |
| 19 | 0.579664 | 0.561295 |

### CPU
Unfortunitely for the model i'm using some operations only support nhwc on device type cpu