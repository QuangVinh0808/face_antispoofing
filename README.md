# FFT Anti-Spoofing

Nghiên cứu nhận diện giả mạo khuôn mặt chéo miền trên **MSU-MFSD** và **SiW-Mv2**, kết hợp đặc trưng RGB, đặc trưng Fourier và tăng cường dữ liệu bằng DDPM sinh biên độ phổ.

Dự án đã triển khai tiền xử lý video, protocol chống rò rỉ dữ liệu, chuỗi ablation E0–E6, benchmark với các phương pháp đối chiếu, đo độ phức tạp suy luận và chẩn đoán chất lượng phổ/ảnh tổng hợp. Phần DDPM đã bổ sung chuẩn hóa biên độ theo bán kính và giới hạn dự đoán biên độ sạch trong sampling.

**Trạng thái thực nghiệm:** các bảng benchmark bên dưới là kết quả đã báo cáo. Các thay đổi về biểu diễn biên độ và sampling được đánh giá riêng bằng diagnostics; chưa có kết quả benchmark classifier mới chứng minh lợi ích của các thay đổi này. Ảnh sinh trong kiểm tra gần nhất vẫn bị biến dạng màu và chưa đạt quality gate.

## 1. Các thành phần đã triển khai

| Thành phần | Nội dung |
|---|---|
| Tiền xử lý | Lấy frame từ video, phát hiện mặt bằng MTCNN, mở rộng vùng mặt và lưu PNG |
| Manifest và QA | Lưu nhãn, video, split, đường dẫn; kiểm tra trùng lặp, đường dẫn thiếu và overlap |
| Cross-dataset protocol | MSU-MFSD → SiW-Mv2 và SiW-Mv2 → MSU-MFSD |
| Classifier | Nhánh RGB, nhánh FFT, fusion, patch stream, contrastive loss và center loss |
| DDPM | Học biên độ Fourier của source-train attack, ghép với pha source-train live để tái tạo ảnh |
| Biểu diễn biên độ | Chuẩn hóa toàn cục, chuẩn hóa log theo bán kính, chuẩn hóa Q95 + căn bậc hai theo bán kính |
| Sampling | Respacing giữ các marginal của lịch nhiễu gốc; giới hạn dự đoán mẫu sạch tại từng bước |
| Kiểm soát chất lượng | Kiểm tra RGB vượt biên, phổ, độ đa dạng, acceptance và fallback về attack thực |
| Báo cáo | Metrics mức video, kết quả theo seed, ablation, Params và Conv/MatMul GFLOPs |

## 2. Dữ liệu và tiền xử lý

Hai bộ dữ liệu đang dùng trong benchmark là MSU-MFSD và SiW-Mv2 Protocol I.

Pipeline đã triển khai:

1. Lấy đều tối đa 32 frame trên mỗi video.
2. Phát hiện khuôn mặt bằng MTCNN.
3. Mở rộng vùng mặt với hệ số context 1,3.
4. Resize và lưu ảnh PNG 256 × 256.
5. Xây dựng manifest và kiểm tra tính hợp lệ của protocol.
6. Chuẩn bị đầu vào mô hình 224 × 224.

Source-train dùng tối đa 16 frame/video. Source-dev và target-test giữ các frame hợp lệ trong tập frame đã trích. Nhãn trong manifest: `0 = live`, `1 = attack`.

| Chiều đánh giá | Source-train | Source-dev | Target-test |
|---|---|---|---|
| `M_to_S` | MSU: 96 video, 1.536 frame | MSU: 24 video, 768 frame | SiW: 623 video, 19.751 frame |
| `S_to_M` | SiW: 845 video, 13.520 frame | SiW: 212 video, 6.740 frame | MSU: 160 video, 5.120 frame |

Đây là số lượng trong manifest benchmark đã báo cáo, sau tiền xử lý và kiểm tra dữ liệu. Thống kê của manifest phải được giữ cùng kết quả chạy để nhận diện đúng tập dữ liệu thực tế.

## 3. Quy tắc đánh giá đã áp dụng

- Chia source-train/source-dev với holdout 20% và kiểm tra overlap theo metadata của protocol.
- Chỉ học trọng số và tính thống kê FFT từ source-train.
- Chỉ dùng source-dev để chọn checkpoint và threshold.
- Chọn checkpoint có ACER thấp nhất trên source-dev.
- Chọn threshold tại điểm EER trên source-dev rồi giữ cố định khi đánh giá target-test.
- Tính score video bằng trung bình score của các frame hợp lệ.
- Target-test không tham gia huấn luyện, fit normalization, chọn checkpoint, threshold hoặc siêu tham số.

Thống kê dùng cho nhánh FFT classifier và thống kê dùng cho DDPM được quản lý riêng. Thống kê biên độ DDPM chỉ được fit trên **source-train attack**.

## 4. Kiến trúc và ablation

Classifier kết hợp đặc trưng không gian từ RGB với đặc trưng tần số từ log biên độ Fourier. Patch stream bổ sung đặc trưng cục bộ trong huấn luyện. Các biến thể dùng cùng protocol để đánh giá tác động của từng thành phần.

| Biến thể | RGB | FFT | Patch | Contrastive | Center | DDPM |
|---|:---:|:---:|:---:|:---:|:---:|:---:|
| E0 | ✓ | — | — | — | — | — |
| E1 | — | ✓ | — | — | — | — |
| E2 | ✓ | ✓ | — | — | — | — |
| E3 | ✓ | ✓ | ✓ | — | — | — |
| E4 | ✓ | ✓ | ✓ | ✓ | — | — |
| E5 | ✓ | ✓ | ✓ | ✓ | ✓ | — |
| E6 | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |

- **E0:** baseline RGB-only với cross-entropy.
- **E1:** baseline FFT-only với cross-entropy.
- **E2–E5:** lần lượt bổ sung fusion RGB/FFT, patch stream, supervised contrastive loss và center loss.
- **E6:** classifier đầy đủ kết hợp tăng cường attack bằng DDPM.

DDPM được pretrain riêng trên source-train attack và đóng băng khi huấn luyện classifier. Khi suy luận, dùng ảnh full-face qua classifier; DDPM và patch stream huấn luyện không nằm trong đường suy luận.

## 5. Sinh ảnh bằng biên độ Fourier

DDPM sinh biên độ của phổ Fourier, không sinh trực tiếp ảnh RGB. Với biên độ sinh `A_DDPM` và pha của ảnh live `phi_live`, ảnh tổng hợp được tái tạo theo:

```text
A_syn = A_DDPM
x_syn = IFFT(A_syn × exp(i × phi_live))
```

Biên độ sinh được chiếu về đối xứng phù hợp với phổ Fourier của ảnh thực trước khi ghép pha. Ảnh sau IFFT được kiểm tra chất lượng; ảnh bị từ chối có cơ chế thay bằng attack thực.

Batch classifier giữ kích thước 16 và tỷ lệ 8 live : 8 attack. Khi augmentation hoạt động với tỷ lệ 0,5, cơ cấu yêu cầu là 8 live + 4 real attack + 4 synthetic attack. Số ảnh synthetic thực sự được dùng phụ thuộc quality gate và fallback.

Benchmark đã báo cáo dùng warm-up 300 classifier steps. Runner pretrain riêng hiện tại cấu hình `warmup_steps = 0`; ngân sách và sampling policy của từng lần chạy được lưu trong config. Không dùng một lịch huấn luyện để mô tả tất cả checkpoint.

## 6. Chuẩn hóa biên độ theo bán kính

Đã triển khai ba chế độ biểu diễn:

| Chế độ | Cách biểu diễn |
|---|---|
| `global_log_standard` | Chuẩn hóa `log(1 + A)` bằng thống kê toàn cục |
| `radial_log_standard` | Chuẩn hóa `log(1 + A)` theo kênh màu và vành tần số |
| `radial_q95_sqrt` | Chia biên độ cho Q95 theo kênh/vành, lấy căn bậc hai và ánh xạ về miền giới hạn |

Với `radial_q95_sqrt`, gọi `s[c,b]` là ước lượng phân vị 95% của biên độ trong kênh `c`, vành `b`. Gọi `u` là ngưỡng robust lấy từ phân vị 99,5% của biên độ đã chia thang và lấy căn bậc hai. Phép mã hóa và giải mã đã triển khai là:

```text
z = 2 × clip(sqrt(A / s[c,b]) / u, 0, 1) − 1
A_hat = s[c,b] × [u × (clip(z, −1, 1) + 1) / 2]²
```

Các hệ số đều được fit từ source-train attack. Q95 là ước lượng bằng lấy mẫu hệ số có seed, không phải phân vị chính xác của toàn bộ hệ số Fourier. Chính sách fit và fingerprint manifest được lưu trong metadata.

Lưới bán kính được tính từ `fftfreq` sau `fftshift`: DC ở tâm có bán kính 0, góc Nyquist có bán kính 1. Đã triển khai tùy chọn radius conditioning bằng một kênh bán kính sạch bổ sung vào UNet. Kênh này không bị diffusion và không phải đầu ra. Trong kiểm tra ảnh gần nhất, **radius conditioning tắt**.

Checkpoint được kiểm tra tính tương thích với mode, hash thống kê và trạng thái radius conditioning. Không tự động dùng checkpoint của một biểu diễn cho biểu diễn khác.

## 7. Ổn định sampling và roundtrip

### Sampling

Đã bổ sung:

- Respacing giữ tích lũy alpha tại các timestep được chọn của lịch nhiễu gốc.
- Giới hạn dự đoán mẫu sạch `x0` trong miền biểu diễn `[-1, 1]` tại từng bước reverse sampling.
- Ghi chỉ số vượt biên trước khi giới hạn để tránh che mất lỗi dự đoán.
- Đối chứng bật/tắt giới hạn trên cùng checkpoint.

Giới hạn mẫu sạch trong sampling, giới hạn khi giải mã biên độ và giới hạn pixel RGB sau IFFT là các phép xử lý ở những vị trí khác nhau. Trạng thái nhiễu `x_t` không bị giới hạn như mẫu sạch.

Trong tên output `checkpoint_on` và `checkpoint_off`, `on/off` biểu thị bật/tắt giới hạn dự đoán biên độ sạch. Chúng không biểu thị hai checkpoint đã huấn luyện độc lập.

### Roundtrip

Roundtrip kiểm tra mã hóa rồi giải mã biên độ của dữ liệu thực, qua đó phân biệt sai số của phép biến đổi với ảnh hưởng của clipping. Ảnh roundtrip không phải ảnh do DDPM sinh.

Một lần kiểm tra đã ghi nhận:

```text
exact_max       = 7.88e-08
bounded_mean    = 0.00525
coefficient_clip = 0.00551
```

Sai số cực đại của phép khứ hồi không clip gần mức sai số số học. Phép giới hạn vẫn gây mất thông tin; các chỉ số này không chứng minh DDPM đã học tốt phân bố attack.

## 8. Diagnostics và kết quả ảnh sinh

Đã triển khai kiểm tra trên source-train, gồm phổ công suất theo bán kính, bản đồ log biên độ, năng lượng tần số cao, RGB vượt biên, khoảng cách phổ, MMD, khoảng cách Wasserstein theo dải và độ đa dạng.

Đối chứng real–real dùng các nhóm attack không chung video. Điều này không tự động có nghĩa là không chung danh tính. Các khoảng cách phổ là diagnostics mô tả; không tự động thay thế threshold của quality gate.

| Output | Nội dung |
|---|---|
| `representation_roundtrip.json` | Sai số khứ hồi và ảnh hưởng của giới hạn biểu diễn |
| `representation_roundtrip_grid.png` | Attack gốc, tái tạo không clip và tái tạo có clip |
| `ddpm_normalization.json` | Thống kê biểu diễn biên độ source-train attack |
| `pretrain_config.yaml` | Config DDPM đã resolve |
| `sampling_k_selection.json` | Metrics từng K và trạng thái chất lượng |
| `live_vs_synthetic_k*.png` | Cặp ảnh live gốc và ảnh synthetic sau IFFT/clamp |
| `mean_spectra_k*.png` | Phổ live, attack thực, DDPM trước IFFT và synthetic sau clamp |
| `radial_power_profiles_k*.png` | Phổ công suất trung bình theo bán kính |
| `spectral_profiles_k*.npz` | Dữ liệu số của radial profiles |

**Cách đọc radial profiles:** trục X là bán kính tần số chuẩn hóa; trục Y là `log(1 + công suất trung bình/vành)`. Các đường biểu diễn live, real spoof, DDPM trước IFFT và synthetic sau pixel clamp. Vùng tô biểu diễn độ lệch chuẩn của các mẫu, không phải khoảng tin cậy.

**Cách đọc bản đồ phổ:** hai chiều là tần số ngang/dọc sau dịch tâm, tần số thấp nằm ở giữa; màu sáng biểu thị biên độ lớn. Các nhóm trong cùng hình dùng chung thang màu. Hai hình có thang màu khác nhau không thể so trực tiếp độ sáng tuyệt đối.

### Kết quả kiểm tra gần nhất

Sau thay đổi biểu diễn và sampling, phổ sinh đã tập trung ở tần số thấp và giảm theo bán kính rõ hơn. Tuy nhiên, vùng tần số thấp vẫn lệch so với attack thực; ảnh RGB còn các mảng màu lớn và bão hòa.

Đối chứng cùng checkpoint với `K = 20`, đánh giá 8 mẫu:

| Chỉ số | Tắt giới hạn mẫu sạch | Bật giới hạn mẫu sạch |
|---|---:|---:|
| Tỷ lệ giá trị RGB vượt miền hợp lệ trước pixel clamp | 63,24% | 33,44% |
| Số mẫu đạt quality gate | 0/8 | 0/8 |

`K = 20` là số bước sampling, không phải số bước huấn luyện. Đây là kiểm tra chất lượng generator trên một nhóm mẫu nhỏ, không phải kết quả benchmark target-test.

Đã khởi chạy kiểm tra tiny-overfit với ngân sách 500 update, nhưng lần chạy được ghi nhận dừng do OOM trước update đầu tiên. Chưa có kết quả hoàn thành 500 update từ lần chạy đó.

Khi tất cả K ứng viên bị quality gate từ chối, runner ghi trạng thái `all_candidates_rejected` và chặn huấn luyện classifier với generator này. K được xếp hạng để chẩn đoán không đồng nghĩa với K đã đạt chất lượng.

## 9. Thiết lập benchmark đã báo cáo

| Tham số | Giá trị |
|---|---|
| Input classifier | 224 × 224 |
| Batch | 16, cân bằng 8 live : 8 attack |
| Optimizer classifier | AdamW |
| Learning rate classifier | 1e-4 |
| Learning rate DDPM trong thiết lập đã báo cáo | 2e-4 |
| Ngân sách classifier | Tối đa 3.000 steps |
| Đánh giá source-dev | Mỗi 300 steps |
| Early stopping | Từ 1.200 steps, patience 4, min delta 0,001 |
| Seeds | 42, 43, 44 |
| Sampling K trong benchmark đã báo cáo | 20 |
| Checkpoint / threshold | Source-dev ACER thấp nhất / source-dev EER |

Chuỗi E0–E6 gồm 42 lần chạy: 2 hướng × 7 biến thể × 3 seeds. Các bảng sau dùng **mean ± sample standard deviation**, `n = 3`, theo báo cáo thực nghiệm. Runner dùng `pstdev` cho CSV tổng hợp; vì vậy CSV tự sinh và bảng dùng sample standard deviation phải được phân biệt khi trích số liệu.

### Ablation trên target-test

HTER và EER tính theo phần trăm; AUC nằm trong `[0, 1]`.

| Chiều | Biến thể | HTER (%) ↓ | AUC ↑ | EER (%) ↓ |
|---|---|---:|---:|---:|
| M → S | E0 | 40.12 ± 1.97 | 0.777 ± 0.006 | 30.28 ± 1.06 |
| M → S | E1 | 47.80 ± 0.24 | 0.523 ± 0.011 | 48.42 ± 1.06 |
| M → S | E2 | 40.60 ± 0.38 | 0.678 ± 0.010 | 37.93 ± 1.92 |
| M → S | E3 | 43.32 ± 1.69 | 0.745 ± 0.032 | 30.88 ± 2.39 |
| M → S | E4 | 42.53 ± 1.66 | 0.735 ± 0.026 | 32.10 ± 1.29 |
| M → S | E5 | 43.46 ± 0.99 | 0.752 ± 0.028 | 31.41 ± 2.43 |
| M → S | E6 | 40.75 ± 2.25 | 0.712 ± 0.027 | 33.71 ± 3.35 |
| S → M | E0 | 20.14 ± 2.08 | 0.888 ± 0.010 | 20.28 ± 2.05 |
| S → M | E1 | 41.25 ± 3.25 | 0.649 ± 0.013 | 39.17 ± 1.18 |
| S → M | E2 | 24.17 ± 1.80 | 0.852 ± 0.009 | 26.53 ± 1.09 |
| S → M | E3 | 20.83 ± 1.56 | 0.907 ± 0.018 | 18.33 ± 3.12 |
| S → M | E4 | 24.17 ± 0.34 | 0.874 ± 0.009 | 23.33 ± 0.90 |
| S → M | E5 | 21.11 ± 6.01 | 0.892 ± 0.026 | 18.33 ± 3.12 |
| S → M | E6 | 20.97 ± 3.05 | 0.868 ± 0.052 | 21.67 ± 4.71 |

RGB-only đạt HTER trung bình thấp nhất trong chuỗi ablation ở cả hai hướng. So với E5, E6 giảm HTER trung bình 2,71 điểm phần trăm ở M → S và 0,14 điểm phần trăm ở S → M, nhưng AUC giảm và EER tăng ở cả hai hướng. Các kết quả này không cho thấy lợi ích đồng đều trên mọi chỉ số, cũng chưa xác lập ý nghĩa thống kê của chênh lệch.

### Các phương pháp đối chiếu

PatchNet dùng biến thể type-only; FSDA dùng thiết lập single-source FSDA-S1. Đây là kết quả chạy theo protocol chung của dự án, không phải số liệu chép từ bài báo gốc.

| Chiều | Phương pháp | HTER (%) ↓ | AUC ↑ | EER (%) ↓ |
|---|---|---:|---:|---:|
| M → S | PatchNet-type-only | 42.34 ± 2.74 | 0.773 ± 0.021 | 30.55 ± 1.76 |
| M → S | FSDA-S1 | 44.82 ± 0.43 | 0.580 ± 0.014 | 46.07 ± 1.85 |
| M → S | DiffFAS | 40.78 ± 3.24 | 0.654 ± 0.023 | 40.34 ± 1.54 |
| S → M | PatchNet-type-only | 21.11 ± 4.59 | 0.892 ± 0.030 | 20.69 ± 6.07 |
| S → M | FSDA-S1 | 28.61 ± 6.74 | 0.786 ± 0.096 | 27.08 ± 7.64 |

DiffFAS đã có kết quả M → S với 3 seeds. Chưa có kết quả S → M do thiếu metadata camera và ghép cặp live–spoof cần cho thiết lập đã dùng.

## 10. Độ phức tạp suy luận

| Phương pháp | Params M → S (M) | Params S → M (M) | GFLOPs/frame M → S | GFLOPs/frame S → M |
|---|---:|---:|---:|---:|
| PatchNet-type-only | 11.1786 | 11.1832 | 16.6552 | 16.6553 |
| FSDA-S1 | 3.6155 | 3.6155 | 72.1384 | 72.1384 |
| DiffFAS | 11.1806 | — | 3.6271 | — |
| E5 | 23.2728 | 23.2728 | 7.2561 | 7.2561 |
| E6 | 23.2728 | 23.2728 | 7.2561 | 7.2561 |

Params là tham số thuộc đường suy luận. GFLOPs ở đây chỉ tính **Conv/MatMul**, với quy ước `1 MAC = 2 FLOPs`, trên một face frame đầu vào. Chưa bao gồm FFT/IFFT, normalization, activation, pooling, softmax, tiền xử lý hoặc tổng hợp score video. FFT vẫn được thực hiện khi suy luận với nhánh FFT; việc không nằm trong số đo này không có nghĩa là chi phí FFT bằng 0.

E5 và E6 có cùng chi phí classifier trong phép đo này vì DDPM chỉ phục vụ tăng cường dữ liệu khi huấn luyện. Các số GFLOPs trên không phải chi phí end-to-end của toàn hệ thống.

## 11. Công cụ chạy đã triển khai

Mã nguồn Python được tổ chức dưới `src/improved_ddpm`; config thí nghiệm nằm trong `configs`. FFmpeg và ffprobe phục vụ đọc/kiểm tra video; PyTorch phục vụ mô hình và FFT; MTCNN phục vụ phát hiện khuôn mặt.

Cài package từ thư mục gốc của repository:

```bash
python -m pip install -r requirements.txt
python -m pip install -e .
```

Các entry point đã triển khai:

| Entry point | Chức năng |
|---|---|
| `prepare_siwmv2_benchmark.py` | Chuẩn bị dữ liệu SiW-Mv2 và ánh xạ protocol |
| `improved_ddpm.scripts.prepare_cross_dataset_ms` | Tạo manifest, split và thống kê cho hai chiều benchmark |
| `improved_ddpm.scripts.run_cross_dataset_ms` | Chạy ablation hoặc DDPM pretrain-only |
| `improved_ddpm.scripts.prepare_ddpm_representation` | Fit biểu diễn biên độ và kiểm tra roundtrip |
| `improved_ddpm.scripts.pretrain_amplitude_ddpm` | Pretrain DDPM biên độ |
| `improved_ddpm.scripts.evaluate_ddpm_sampling_k` | Đánh giá chất lượng tại các số bước sampling |
| `improved_ddpm.scripts.evaluate_synthetic_quality` | Kiểm tra ảnh tổng hợp |
| `improved_ddpm.scripts.train` | Huấn luyện classifier |
| `improved_ddpm.scripts.evaluate_protocol_a` | Đánh giá với checkpoint và protocol đã khóa |

Ví dụ cấu hình diagnostic đã dùng: source MSU, seed 42, 32 attack images, 20 pretrain updates, radial Q95 + sqrt và radius conditioning tắt. Lệnh tương ứng, với một thư mục output riêng:

```bash
python -m improved_ddpm.scripts.run_cross_dataset_ms \
  --base-config configs/ms_e0_e6.yaml \
  --benchmark-root data/benchmarks/cross_dataset_ms \
  --settings M_to_S --variants E6 --seeds 42 \
  --pretrain-only --ddpm-tiny-count 32 \
  --ddpm-pretrain-steps 20 \
  --ddpm-representation radial_q95_sqrt \
  --radius-conditioning off \
  --sampling-k-candidates 20 50 100 \
  --sampling-quality-count 8 \
  --output-root data/experiments/ddpm_diagnostics
```

Đây là cấu hình kiểm tra pipeline, không phải cấu hình đã chứng minh hội tụ. `--pretrain-only` bỏ qua huấn luyện/đánh giá classifier. Thay mode, thống kê, radius hoặc ngân sách cần output riêng để tránh dùng nhầm cache/checkpoint.

Đã bổ sung kiểm tra cho biểu diễn radial và bounded sampling, gồm inverse, tính tương thích checkpoint, bảo toàn marginal khi respacing và giới hạn trên mẫu sạch thay vì trạng thái nhiễu. Các file kiểm tra tương ứng:

```text
tests/test_ddpm_radial_representation.py
tests/test_ddpm_bounded_sampling.py
```

Kiểm tra số học bằng fixture không thay thế đánh giá chất lượng generator trên dữ liệu thực.

## 12. Kết luận từ phần việc đã thực hiện

- Đã xây dựng benchmark chéo miền và hoàn thành chuỗi ablation E0–E6 theo thiết lập được báo cáo.
- Đã đối chiếu với PatchNet, FSDA-S1 và DiffFAS trong phạm vi có đủ dữ liệu/metadata.
- Đã xác định sai lệch phân bố phổ và tình trạng RGB vượt miền hợp lệ của ảnh DDPM.
- Đã triển khai chuẩn hóa Q95 + sqrt theo bán kính, kiểm tra roundtrip và ổn định sampling bằng giới hạn mẫu sạch.
- Phổ sinh và tỷ lệ RGB vượt biên đã cải thiện trong diagnostics, nhưng nhóm mẫu kiểm tra gần nhất vẫn có acceptance 0/8 và biến dạng màu rõ rệt.
- Chưa có bằng chứng benchmark mới cho thấy các thay đổi generator này cải thiện khả năng tổng quát hóa của classifier.
