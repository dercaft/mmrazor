#  项目记录
## 项目文件结构
``` shell

```
## 记录疑问与问题
### Question
``` shell
config.py:
    resume_from
    load_from
<main>.py:
    checkpoint
```
差别是什么？都是在什么时候加载的？会互相覆盖吗？
### BUG

Autoslim和Hybrid都报错：

```
super(RatioPruner, self).prepare_from_supernet(supernet)

self.trace_norm_conv_links(pseudo_loss.grad_fn, module2name,

File "/home/wyh/wuyuhang/mmrazor/mmrazor/models/pruners/structure_pruning.py", line 710, in trace_norm_conv_links
    conv_grad_fn = conv_grad_fn.next_functions[0][0]
```
循环到conv_grad_fn为None后，还在继续循环

``` shell
AttributeError: AutoSlim: 'NoneType' object has no attribute 'next_functions'
```
## 代办事项

1. mmseg config&checkpoint
2. mmcls resnet/mobilenetv2 checkpoint