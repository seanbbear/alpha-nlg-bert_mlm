# alpha-nlg-bert_mlm

## Dataset
[Abductive Commonsense Reasoning](https://openreview.net/pdf?id=Byg1v1HKDB)
在 ICLR 2020 上發表

## Definition
*O1* : 在時間t1的觀測現象
*O2* : 在時間t2(t2>t1)的觀測現象
*H+* : 合理的假設，解釋了O1及O2 
*H-* : 不合理的假設

## Data Example
```
{
    "story_id": "00050cbb-049e-444f-a17a-04c882da4693-1", 
    "obs1": "Chad went to get the wheel alignment measured on his car.", 
    "obs2": "The mechanic provided a working alignment with new body work.",     
    "hyp1": "Chad was waiting for his car to be washed.", 
    "hyp2": "Chad was waiting for his car to be finished."
}
```

