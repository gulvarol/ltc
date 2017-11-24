This directory contains the iDT+FV scores for UCF101 and HMDB51 for each of their splits. If you do the following, you get the accuracy for one split:

```
load('ucf_idt_split1.mat');
[~, pred] = max(scores_idt');
acc = mean(pred' == gt_idt);
```

As stated in the paper: "Our implementation of IDT+FV [5] obtained 84.5% and 57.3% for UCF101 and HMDB51, respectively."
