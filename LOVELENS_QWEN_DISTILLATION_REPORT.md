# 訊息分析 Qwen 小模型蒸餾評估報告

## 一句話結論

可以做，而且值得做。這個 app 的任務非常特化，所以最適合的不是單純「壓縮 Qwen」，而是做「訊息分析報告 task distillation」：用 Qwen 1.7B 當老師，訓一個 Qwen3 0.6B 學生，專門輸出你的手機報告格式。

目標是把目前 1.0GB 的 1.7B GGUF，降到約 350-500MB 等級，同時讓輸出更穩、更像你的報告。

## 我做了什麼

我已經在專案裡建立第一版蒸餾包：

- `distillation/lovelens_text_qwen_student_v1/schema.json`
- `distillation/lovelens_text_qwen_student_v1/seed_teacher_examples.jsonl`
- `distillation/lovelens_text_qwen_student_v1/README.md`

這份包定義了學生模型要學的 19 個報告欄位、seed examples、資料量、訓練主線、評估門檻與手機大小預估。

## 為什麼小模型可能追上甚至超過 1.7B

Qwen 1.7B 是通用模型；它會聊天、寫作、推理，但你的 app 不需要全部能力。你的 app 只需要：

- 讀兩人聊天紀錄。
- 判斷關係類型。
- 判斷兩人的角色與互動模式。
- 輸出優點、盲點、相性判讀、改善句。
- 固定成繁中、直白、溫暖、手機可讀的報告。

小模型如果只針對這件事訓練，可能在產品指標上贏：

- JSON 更穩。
- 報告更一致。
- 少講廢話。
- 更少跑題。
- 速度更快。
- app 體積更小。

但它不會在通用推理上贏 1.7B。這是「產品任務變強」，不是「全能力變強」。

## 模型建議

| 排名 | 模型 | 建議定位 | 判斷 |
|---:|---|---|---|
| 1 | Qwen3 0.6B Q4/Q5 | 第一個學生模型 | 最適合先試，跟 1.7B 同家族，任務夠窄 |
| 2 | Qwen3 1.7B Q4 | teacher / benchmark | 保留當老師，不建議當長期唯一 app 模型 |
| 3 | Qwen2.5 0.5B Instruct | 超小備選 | 可以試，但中文關係報告深度要實測 |
| 4 | SmolLM2 360M Instruct | 極小實驗 | 可拿來測「規則分析 + 小模型寫報告」 |

我不建議一開始就追更小。先用 0.6B 打到好用，再往下砍。

## 0.6B 能不能再蒸更小

可以，但最穩的方式不是把 0.6B 直接硬壓成更小，而是讓 0.6B/1.7B 當老師，去教一個更小的學生模型。

我會分三條線：

| 路線 | 目標大小 | 建議 |
|---|---:|---|
| 0.6B LoRA + Q4/Q5 | 約 350-650MB | 最適合第一個產品版本 |
| Qwen2.5 0.5B student | 約 300-450MB | Qwen 家族內再小一點，值得實測 |
| 360M / 135M student | 約 80-300MB | 實驗線，需靠 app 規則先抽訊號 |

我的判斷：如果你要「報告像樣、角色判斷不太蠢、中文口氣穩」，0.6B 是比較好的第一刀。若目標是 app 很小，才做 0.5B 或 360M，但 app 架構要改成「規則先分析，小模型只負責寫報告」。

## 預估 app 大小

| 版本 | 模型 | 預估 app 大小 |
|---|---|---:|
| 現在 | Qwen3 1.7B Q4_K_S | 約 1.0GB 實測 |
| 蒸餾 v1 | Qwen3 0.6B Q4_K_M | 約 350-500MB |
| 蒸餾 v1 品質版 | Qwen3 0.6B Q5_K_M | 約 450-650MB |
| 極小版 | 規則分析 + 小模型只改寫 | 約 300-500MB 或更低 |

精確大小要下載/匯出 GGUF 後量測。

## 最實際做法

1. App 先解析 LINE/IG/Threads/Facebook 匯出檔。
2. 用規則抽訊號：回覆速度、打斷、道歉、責備、照顧、修復句。
3. Qwen 1.7B teacher 產生標準 JSON 報告。
4. 過濾壞資料：JSON 壞掉、欄位缺漏、太像算命、過度診斷。
5. 用留下的資料 SFT/LoRA Qwen3 0.6B。
6. 匯出 GGUF Q4/Q5。
7. 放進 iPhone app 實測。

## 成功標準

我會用這些標準決定是否換掉 1.7B：

- JSON 可解析率 >= 98%
- 19 個欄位完整率 >= 97%
- 關係類型判斷 >= 85%
- 改寫句可用率 >= 90%
- 危險或過度診斷語句 <= 1%
- iPhone 真機生成速度比 1.7B 快 30% 以上
- 模型大小比 1.7B 小 45% 以上

## 下一步

如果要真的做訓練，我建議直接跑 v0：

- 先產 500 筆 teacher data。
- 訓 Qwen3 0.6B LoRA。
- 用 100 筆 gold set 評估。
- 匯出 Q4 GGUF。
- 替換 app 內模型，量真機速度和 app 大小。

這會比繼續調 prompt 更有價值，因為它會直接回答：0.6B 能不能接住「訊息分析」這件事。

## 來源

- Qwen3 0.6B official model card: https://huggingface.co/Qwen/Qwen3-0.6B
- Qwen3 1.7B official model card: https://huggingface.co/Qwen/Qwen3-1.7B
- Qwen2.5 0.5B official model card: https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct
- SmolLM2 collection: https://huggingface.co/collections/HuggingFaceTB/smollm2-6723884218bcda64b34d7db9
- 本機實測：`LoveLensDemo/Models/Qwen3_1.7B.Q4_K_S.gguf` 約 1.0GB
