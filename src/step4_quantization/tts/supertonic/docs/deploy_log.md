# Nhat ky deploy len chip (AI Hub, Dragonwing IQ-9075 EVK)

Moi dong la mot thi nghiem; job ID xem duoc tai `https://workbench.aihub.qualcomm.com/jobs/<ID>/` (tai khoan AI Hub cua tac gia).
Log da duoc loc bo thanh tien trinh; ban goc nam trong `logs/chip/`. Cac job `ve_c*_*`, `ve_cut*`, `syn_t*` la graph cat nho dung de bisect (xem docs/bug_reducesum.md).

## Thi nghiem chinh

| Log | Mo ta | Job ID (loai:ID) | Ket qua |
|---|---|---|---|
| `hub_full_dbg.log` | Graph gop + dau ra trung gian (text_emb, xt1, xt5, dur) | compile: `jgk69vjv5`<br>profile: `jpx0d9w8p`<br>infer: `j5m9dej7g` | full_dbg infer SUCCESS wall 1354s<br>full_dbg saved ['output_0', 'output_1', 'output_2', 'output_3', 'output_4', 'output_5'] {'output_0': (1, 460800), 'output_1': (1,), 'output_2': (1, 25<br>full_dbg profile SUCCESS |
| `hub_full_fp16.log` | Graph gop toan bo (fp16) - lan dau, chat luong HONG | compile: `jpe6812og`<br>profile: `jprxoqxkp`<br>infer: `jp2o46o6g` | full_fp16 infer SUCCESS wall 1406s<br>full_fp16 saved ['output_0', 'output_1'] {'output_0': (1, 460800), 'output_1': (1,)}<br>full_fp16 profile SUCCESS |
| `hub_full_v2.log` | Graph gop, bo nhanh thoi gian (gia thuyet sai) | compile: `j56odzn75`<br>profile: `jgn17m9mp`<br>infer: `jprxn24ep` | full_v2 infer SUCCESS wall 1646s<br>full_v2 saved ['output_0', 'output_1'] {'output_0': (1, 460800), 'output_1': (1,)}<br>full_v2 profile SUCCESS |
| `hub_full_v3.log` | **Graph gop DA SUA ReduceSum - ket qua cuoi** | compile: `jprxn9nep`<br>profile: `jpe6l2l0g`<br>infer: `jgzz4w46g` | full_v3 infer SUCCESS wall 1496s<br>full_v3 saved ['output_0', 'output_1'] {'output_0': (1, 460800), 'output_1': (1,)}<br>full_v3 profile SUCCESS |
| `rotB_cut.log` | Rotary: Unsqueeze (cat Concat_3) | compile: `jgj3wko8p` | infer SUCCESS<br>ve_rotB_cut_L150 step 0: Concat_3 valid-frame cos 0.21138<br>ve_rotB_cut_L150 step 4: Concat_3 valid-frame cos 0.35921 |
| `rotB_full.log` | Rotary: Unsqueeze (VE day du) | compile: `jpe6l480g` | infer SUCCESS<br>ve_rotB_L150 ONE-STEP cos per ODE step: {0: 0.99765, 1: 0.9775, 2: 0.95602, 3: 0.94806, 4: 0.9407} |
| `ve_erftanh.log` | VE: thay Erf bang tanh | compile: `jgod4knd5`<br>infer: `jpx0dny9p` | erftanh infer SUCCESS 574s<br>erftanh ONE-STEP cos vs CPU fp32 per ODE step: {0: 0.99765, 1: 0.97858, 2: 0.95608, 3: 0.94849, 4: 0.9409} |
| `ve_finite.log` | VE: thay -inf bang -30000 | compile: `jpe6lk40g`<br>infer: `jg9oxw8lg` | finite infer SUCCESS 508s<br>finite ONE-STEP cos vs CPU fp32 per ODE step: {0: 0.99765, 1: 0.9775, 2: 0.95602, 3: 0.94806, 4: 0.9407} |
| `ve_fp16.log` | VE rieng fp16 (L=150) | compile: `jglw162lp`<br>infer: `j57o70xrg` | fp16 infer SUCCESS 474s<br>fp16 ONE-STEP cos vs CPU fp32 per ODE step: {0: 0.99765, 1: 0.9775, 2: 0.95602, 3: 0.94806, 4: 0.9407} |
| `ve_fp16_L120.log` | VE fp16, L=120 | compile: `j5wy1mz6g`<br>infer: `jpx0der3p` | fp16 infer SUCCESS 572s<br>fp16 ONE-STEP cos vs CPU fp32 per ODE step: {0: 0.99767, 1: 0.97681, 2: 0.9585, 3: 0.95753, 4: 0.95775} |
| `ve_fp16_L128.log` | VE fp16, L=128 | compile: `jgzz4vv6g`<br>infer: `jp2ov3jmg` | fp16 infer SUCCESS 447s<br>fp16 ONE-STEP cos vs CPU fp32 per ODE step: {0: 0.99759, 1: 0.96512, 2: 0.95899, 3: 0.95852, 4: 0.95821} |
| `ve_fp16_L160.log` | VE fp16, L=160 | compile: `jp1ovqql5`<br>infer: `jpy87vn4g` | fp16 infer SUCCESS 490s<br>fp16 ONE-STEP cos vs CPU fp32 per ODE step: {0: 0.99756, 1: 0.98717, 2: 0.95235, 3: 0.93791, 4: 0.9269} |
| `ve_fp16_L96.log` | VE fp16, L=96 | compile: `jg9ox92lg`<br>infer: `jprxn1d0p` | fp16 infer SUCCESS 464s<br>fp16 ONE-STEP cos vs CPU fp32 per ODE step: {0: 0.99777, 1: 0.96922, 2: 0.95295, 3: 0.95372, 4: 0.94953} |
| `ve_multi.log` | Graph nhieu dau ra: len, Div, Mul, Cos, Concat_3 (PHAT HIEN ReduceSum=1) |  | infer SUCCESS<br>  ReduceSum  shape [2] cpu[62.000,62.000] npu[1.000,1.000] cos 1.00000 first [62.0, 62.0] / [1.0, 1.0]<br>  ReduceSum  shape [2] cpu[62.000,62.000] npu[1.000,1.000] cos 1.00000 first [62.0, 62.0] / [1.0, 1.0] |
| `ve_rotR_L150.log` | **VE thay ReduceSum bang MatMul-voi-vector-1 (SUA DUNG)** | compile: `jgod4me45` | infer SUCCESS<br>ve_rotR_L150 ONE-STEP cos per ODE step: {0: 1.0, 1: 1.0, 2: 0.99999, 3: 0.99999, 4: 0.99997} |
| `ve_rotS_L150.log` | Rotary: thay Slice bang MatMul (VE) | compile: `j5m9d46yg` | infer SUCCESS<br>ve_rotS_L150 ONE-STEP cos per ODE step: {0: 0.99765, 1: 0.9775, 2: 0.95602, 3: 0.94806, 4: 0.9407} |
| `ve_rotS_cut_L150.log` | Rotary: thay Slice bang MatMul (cat) | compile: `jpx0dm6jp` | infer SUCCESS<br>ve_rotS_cut_L150 step 0: Concat_3 valid-frame cos 0.21138<br>ve_rotS_cut_L150 step 4: Concat_3 valid-frame cos 0.35921 |
| `ve_rotT_L150.log` | Rotary: bang cos/sin tinh san (VE) | compile: `j5wy1ml6g` | infer SUCCESS<br>ve_rotT_L150 ONE-STEP cos per ODE step: {0: 0.99764, 1: 0.97177, 2: 0.95607, 3: 0.94753, 4: 0.94027} |
| `ve_rotT_cut_L150.log` | Rotary: bang cos/sin tinh san (cat) | compile: `jg9ox9zlg` | infer SUCCESS<br>ve_rotT_cut_L150 step 0: Concat_3 valid-frame cos 0.20947<br>ve_rotT_cut_L150 step 4: Concat_3 valid-frame cos 0.35890 |
| `ve_tin.log` | VE: nhanh thoi gian tinh san | compile: `j5q4mr3mg`<br>infer: `jprxn2yep` | infer SUCCESS<br>ONE-STEP cos vs CPU fp32 (time_emb precomputed): {0: 0.99766, 1: 0.97738, 2: 0.95606, 3: 0.94811, 4: 0.94073} |
| `ve_w16a16.log` | VE W16A16 (loi compile: thieu bias) | quantize: `jpv290rmg`<br>compile: `jp0ov3lep` | w16a16 compile FAILED QAIRT converter failed with exit code 1 |
| `ve_w16a16zb.log` | VE W16A16 + zero bias | quantize: `jgk6978w5`<br>compile: `jpx0dnl3p`<br>infer: `jp2ov9yxg` | w16a16zb infer SUCCESS 466s<br>w16a16zb ONE-STEP cos vs CPU fp32 per ODE step: {0: 0.99867, 1: 0.98373, 2: 0.9555, 3: 0.91739, 4: 0.74985} |
| `ve_w8a16.log` | VE W8A16 (loi compile: thieu bias) | quantize: `jp3owv1zp`<br>compile: `jpy87r64g` | w8a16 compile FAILED QAIRT converter failed with exit code 1 |
| `ve_w8a16zb.log` | VE W8A16 + zero bias | quantize: `jp8j401k5`<br>compile: `jglw16yjp`<br>infer: `jpy87j3rg` | w8a16zb infer SUCCESS 476s<br>w8a16zb ONE-STEP cos vs CPU fp32 per ODE step: {0: 0.99802, 1: 0.9819, 2: 0.95307, 3: 0.91638, 4: 0.76186} |

## Graph cat nho (bisect)

| Log | Job ID |
|---|---|
| `syn_t1.log` |  |
| `syn_t2.log` |  |
| `syn_t3.log` |  |
| `syn_t4.log` |  |
| `ve_c2_0.log` | compile:`jprxnen9p` |
| `ve_c2_1.log` | compile:`jgd6zozep` |
| `ve_c2_2.log` | compile:`jpx0dyd1p` |
| `ve_c2_3.log` | compile:`jp4e9v9vg` |
| `ve_c2_4.log` | compile:`jgn1737rp` |
| `ve_c2_5.log` | compile:`j57o7x7lg` |
| `ve_c2_6.log` | compile:`jp2ovlv4g` |
| `ve_c2_7.log` | compile:`j5m9d3dwg` |
| `ve_c2_8.log` | compile:`jpy87677g` |
| `ve_c3_0.log` | compile:`j57o7xq9g` |
| `ve_c3_1.log` | compile:`jp1over75` |
| `ve_c3_2.log` | compile:`jpx0dywlp` |
| `ve_c3_3.log` | compile:`jgd6zojzp` |
| `ve_c3_4.log` | compile:`jp4e9vz1g` |
| `ve_c3_5.log` | compile:`j5m9d3j9g` |
| `ve_c4_0.log` | compile:`jgzz4r94g` |
| `ve_c4_1.log` | compile:`jg9oxw1mg` |
| `ve_c4_2.log` | compile:`j5wy1qv4g` |
| `ve_c5_0.log` | compile:`jp0ovr00p` |
| `ve_c5_1.log` | compile:`jgod4v615` |
| `ve_c5_2.log` | compile:`jp3owm0mp` |
| `ve_c5_3.log` | compile:`j56od14n5` |
| `ve_c5_4.log` | compile:`jpy87600g` |
| `ve_c5_5.log` | compile:`jglw1km2p` |
| `ve_c5_6.log` | compile:`jgk69yxv5` |
| `ve_c5_7.log` | compile:`j5q4m2qeg` |
| `ve_c5_8.log` | compile:`jp8j47yq5` |
| `ve_cut0.log` | compile:`jgn1736vp` |
| `ve_cut1.log` | compile:`jp2ovlkxg` |
| `ve_cut2.log` | compile:`jpy8761rg` |
| `ve_cut3.log` | compile:`jp0ovlz2p` |
| `ve_cut4.log` | compile:`j5m9d3xyg` |

## Profile (thoi gian tren chip, do bang `submit_profile_job`)

| Graph | Profile job | Thoi gian/cau | Layer tren NPU | Dinh bo nho | First load |
|---|---|---|---|---|---|
| Gop, fp16 (chat luong hong) | `jprxoqxkp` | 746.1 ms | 4476/4476 | 10 MB | 730 s |
| Gop v2 (bo nhanh thoi gian) | `jgn17m9mp` | 746.4 ms | 4476/4476 | 12 MB | 737 s |
| **Gop v3 (sua ReduceSum)** | `jpe6l2l0g` | **748.3 ms** | **4476/4476** | 10 MB | 740 s |

`first load` la thoi gian nap model lan dau tren thiet bi (khong tinh vao thoi gian suy luan).
