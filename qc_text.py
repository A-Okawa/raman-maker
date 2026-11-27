"""Texts of the QC checks (qc.py) in Japanese, English, Thai and Vietnamese.

``render(item, language, thresholds)`` gives the check's name, its criterion, the reason for the
status and what to look at. The QC report files (JSON, CSV) use English; the panel uses the
interface language. Status words (PASS, CAUTION, WARNING, FAIL, NOT EVALUATED) are not translated.
"""

from __future__ import annotations

import math


def _t(ja, en, th, vi):
    return {"ja": ja, "en": en, "th": th, "vi": vi}


SECTIONS = {
    "input": _t("入力データ", "Input data", "ข้อมูลนำเข้า", "Dữ liệu đầu vào"),
    "processing": _t("前処理", "Processing", "การประมวลผล", "Xử lý"),
    "peaks": _t("ピーク", "Peaks", "พีก", "Đỉnh"),
    "fit": _t("ピーク分離フィット", "Peak fit", "การฟิตแยกพีก", "Khớp tách đỉnh"),
}

PASS_REASON = _t("基準を満たしています。", "Meets the criterion.", "ผ่านเกณฑ์", "Đạt tiêu chí.")

NOT_EVALUATED_REASONS = {
    "smoothed": _t("平滑化したデータでは雑音の推定が成り立たないため評価しません（± の誤差も過小になります）。定量には平滑化を外してフィットしてください。",
                   "Not evaluated on smoothed data: the noise estimate does not hold (the ± errors are too small as well). Fit without smoothing for numbers.",
                   "ไม่ประเมินกับข้อมูลที่ทำให้เรียบแล้ว เพราะการประมาณสัญญาณรบกวนใช้ไม่ได้ (ค่าคลาดเคลื่อน ± ก็เล็กเกินจริง) หากต้องการค่าเชิงปริมาณให้ฟิตโดยไม่ทำให้เรียบ",
                   "Không đánh giá trên dữ liệu đã làm trơn vì ước lượng nhiễu không còn đúng (sai số ± cũng bị nhỏ đi). Để có số liệu, hãy khớp mà không làm trơn."),
    "no_noise": _t("雑音を推定できないため評価できません。", "The noise could not be estimated.",
                   "ประมาณสัญญาณรบกวนไม่ได้", "Không ước lượng được nhiễu."),
    "no_errors": _t("共分散行列が求まらず、誤差・相関を評価できません。", "The covariance matrix could not be obtained, so errors and correlations are not evaluated.",
                    "หาเมทริกซ์ความแปรปรวนร่วมไม่ได้ จึงประเมินค่าคลาดเคลื่อนและสหสัมพันธ์ไม่ได้",
                    "Không tính được ma trận hiệp phương sai nên không đánh giá được sai số và tương quan."),
    "at_limit": _t("上限・下限に到達したパラメータの誤差は意味を持たないため評価しません。", "An error of a parameter on a bound has no meaning; not evaluated.",
                   "ค่าคลาดเคลื่อนของพารามิเตอร์ที่อยู่บนขีดจำกัดไม่มีความหมาย จึงไม่ประเมิน",
                   "Sai số của tham số nằm trên giới hạn không có ý nghĩa nên không đánh giá."),
    "reference": _t("参考値（判定に使いません）。", "Reference value (not used for the judgement).",
                    "ค่าอ้างอิง (ไม่ใช้ในการตัดสิน)", "Giá trị tham khảo (không dùng để đánh giá)."),
    "too_few": _t("点数が少なく評価できません。", "Too few points to evaluate.", "จำนวนจุดน้อยเกินไปที่จะประเมิน", "Quá ít điểm để đánh giá."),
    "": _t("評価していません。", "Not evaluated.", "ไม่ได้ประเมิน", "Chưa đánh giá."),
}

# name, criterion (format string; {…} from the item's params and the thresholds), problem (reason when
# the status is not PASS), info (reason on PASS, optional), advice (what to check when not PASS)
CHECKS = {
    "points": dict(
        name=_t("データ点数", "Data points", "จำนวนจุดข้อมูล", "Số điểm dữ liệu"),
        criterion="≥ 5",
        problem=_t("点数が少なすぎて処理できません。", "Too few points to process.", "จำนวนจุดน้อยเกินไปที่จะประมวลผล", "Quá ít điểm để xử lý."),
        advice=_t("ファイルの中身を確認してください。", "Check the file contents.", "ตรวจสอบเนื้อหาของไฟล์", "Kiểm tra nội dung tệp.")),
    "x_range": dict(
        name=_t("データの範囲", "Data range", "ช่วงของข้อมูล", "Khoảng dữ liệu"), criterion="—"),
    "skipped_lines": dict(
        name=_t("データ途中の読めない行", "Unreadable lines inside the data", "บรรทัดที่อ่านไม่ได้ภายในข้อมูล", "Dòng không đọc được giữa dữ liệu"),
        criterion="0",
        problem=_t("データの途中で {value} 行を読み飛ばしました（行 {lines}：{text}）。その点は図にも解析にも入っていません。",
                   "{value} lines inside the data were skipped (lines {lines}: {text}). Those points are in neither the figure nor the analysis.",
                   "ข้ามไป {value} บรรทัดภายในข้อมูล (บรรทัด {lines}: {text}) จุดเหล่านั้นไม่อยู่ในรูปและการวิเคราะห์",
                   "Đã bỏ qua {value} dòng giữa dữ liệu (dòng {lines}: {text}). Các điểm đó không có trong hình và phân tích."),
        advice=_t("元ファイルの該当行を確認し、必要なら測定ソフトから書き出し直してください。", "Look at those lines in the file and export again if needed.",
                  "ตรวจบรรทัดเหล่านั้นในไฟล์ และส่งออกใหม่หากจำเป็น", "Xem các dòng đó trong tệp và xuất lại nếu cần.")),
    "footer": dict(
        name=_t("データの後ろの行", "Lines after the data", "บรรทัดหลังข้อมูล", "Dòng sau dữ liệu"),
        criterion=_t("数値の行が残っていない", "no numeric rows left unread", "ไม่มีแถวตัวเลขที่ไม่ได้อ่าน", "không còn hàng số chưa đọc"),
        info=_t("データの後ろの {value} 行（行 {lines}）は数値でないため読んでいません。", "The {value} lines after the data (lines {lines}) are not numbers and were not read.",
                "ไม่ได้อ่าน {value} บรรทัดหลังข้อมูล (บรรทัด {lines}) เพราะไม่ใช่ตัวเลข", "{value} dòng sau dữ liệu (dòng {lines}) không phải số nên không được đọc."),
        problem=_t("データの後ろに読まなかった数値の行が {numeric} 行あります（行 {lines}）。x 軸が続かないため別の表と判断しました。",
                   "{numeric} numeric rows after the data were not read (lines {lines}); they do not continue the x axis and were taken as another table.",
                   "มีแถวตัวเลข {numeric} แถวหลังข้อมูลที่ไม่ได้อ่าน (บรรทัด {lines}) เพราะไม่ต่อเนื่องกับแกน x จึงถือเป็นตารางอื่น",
                   "Có {numeric} hàng số sau dữ liệu không được đọc (dòng {lines}); chúng không nối tiếp trục x nên được coi là bảng khác."),
        advice=_t("それが測定データの続きなら、ファイルを整えてから読み込んでください。", "If they belong to the measurement, tidy the file and read it again.",
                  "หากเป็นส่วนต่อของข้อมูลการวัด ให้จัดไฟล์แล้วอ่านใหม่", "Nếu chúng thuộc phép đo, hãy sửa tệp rồi đọc lại.")),
    "cut_rows": dict(
        name=_t("列数の多い行", "Rows with extra columns", "แถวที่มีคอลัมน์เกิน", "Hàng có cột thừa"),
        criterion="0",
        problem=_t("{value} 行で余分な列を無視しました（行 {lines}）。", "Extra columns were ignored in {value} rows (lines {lines}).",
                   "ละคอลัมน์ส่วนเกินใน {value} แถว (บรรทัด {lines})", "Đã bỏ qua cột thừa ở {value} hàng (dòng {lines})."),
        advice=_t("区切り文字や列の並びを確認してください。", "Check the delimiter and the column layout.", "ตรวจตัวคั่นและการจัดคอลัมน์", "Kiểm tra dấu phân cách và bố cục cột.")),
    "nonfinite": dict(
        name=_t("NaN・Inf の点", "NaN / Inf points", "จุด NaN / Inf", "Điểm NaN / Inf"),
        criterion="0",
        problem=_t("数値でない値の {value} 点を除外しました（{where}）。", "{value} points with values that are not finite numbers were left out ({where}).",
                   "ตัดจุดที่ไม่ใช่ตัวเลขจำกัดออก {value} จุด ({where})", "Đã loại {value} điểm có giá trị không hữu hạn ({where})."),
        advice=_t("元ファイルの該当点を確認してください。", "Check those points in the file.", "ตรวจจุดเหล่านั้นในไฟล์", "Kiểm tra các điểm đó trong tệp.")),
    "declared_points": dict(
        name=_t("宣言点数との一致", "Declared point count", "จำนวนจุดที่ระบุในไฟล์", "Số điểm khai báo"),
        criterion=_t("読んだ点数 = NPOINTS", "points read = NPOINTS", "จำนวนที่อ่าน = NPOINTS", "số điểm đọc = NPOINTS"),
        problem=_t("ファイルは {declared} 点と宣言していますが、読めたのは {read} 点です。", "The file declares {declared} points but {read} were read.",
                   "ไฟล์ระบุ {declared} จุด แต่อ่านได้ {read} จุด", "Tệp khai báo {declared} điểm nhưng chỉ đọc được {read} điểm."),
        advice=_t("ファイルが途中で切れていないか確認してください。", "Check that the file is complete.", "ตรวจว่าไฟล์ครบถ้วน", "Kiểm tra tệp có đầy đủ không.")),
    "x_units": dict(
        name=_t("x の単位", "x units", "หน่วยแกน x", "Đơn vị trục x"),
        criterion="cm⁻¹",
        problem=_t("ファイルの単位は {units} です。変換せず cm⁻¹ として描画しています。", "The file says {units}; it is drawn as cm⁻¹ without conversion.",
                   "ไฟล์ระบุหน่วย {units} วาดเป็น cm⁻¹ โดยไม่แปลง", "Tệp ghi {units}; được vẽ theo cm⁻¹ mà không chuyển đổi."),
        advice=_t("単位を確認してください。", "Check the units.", "ตรวจสอบหน่วย", "Kiểm tra đơn vị.")),
    "x_converted": dict(
        name=_t("x の単位変換", "x unit conversion", "การแปลงหน่วยแกน x", "Đổi đơn vị trục x"),
        criterion="—",
        info=_t("{units} から cm⁻¹ に変換しました。", "Converted from {units} to cm⁻¹.", "แปลงจาก {units} เป็น cm⁻¹", "Đã đổi từ {units} sang cm⁻¹.")),
    "duplicates": dict(
        name=_t("同じ波数の点", "Repeated wavenumbers", "เลขคลื่นซ้ำ", "Số sóng trùng lặp"),
        criterion="0",
        problem=_t("{value} 点が直前と同じ波数です（{where} cm⁻¹）。そのまま使っています。", "{value} points repeat the previous wavenumber ({where} cm⁻¹); they are used as they are.",
                   "{value} จุดมีเลขคลื่นซ้ำกับจุดก่อนหน้า ({where} cm⁻¹) ใช้ตามเดิม", "{value} điểm trùng số sóng với điểm trước ({where} cm⁻¹); vẫn được dùng nguyên."),
        advice=_t("複数の測定をつないだデータでないか確認してください。", "Check whether several measurements were joined.",
                  "ตรวจว่าไม่ได้นำการวัดหลายครั้งมาต่อกัน", "Kiểm tra xem có phải nhiều phép đo được nối lại không.")),
    "gaps": dict(
        name=_t("データの抜け", "Missing ranges", "ช่วงข้อมูลที่ขาด", "Khoảng thiếu dữ liệu"),
        criterion="Δx ≤ {gap} × median",
        problem=_t("点間隔が周囲の {gap} 倍を超える箇所が {value} か所あります（{where} cm⁻¹ の直前、間隔の中央値 {median_step}）。",
                   "{value} spacings exceed {gap} × the local spacing (before {where} cm⁻¹; median spacing {median_step}).",
                   "มีระยะห่างระหว่างจุดเกิน {gap} เท่าของบริเวณใกล้เคียง {value} แห่ง (ก่อน {where} cm⁻¹ ค่ามัธยฐาน {median_step})",
                   "Có {value} khoảng cách vượt {gap} lần khoảng cách lân cận (trước {where} cm⁻¹; trung vị {median_step})."),
        advice=_t("抜けをまたぐ平滑化・ベースライン・フィットは、点の番号で計算するため不正確になります。", "Smoothing, baselines and fits across a gap work on point indices and become inaccurate.",
                  "การทำให้เรียบ เบสไลน์ และการฟิตข้ามช่วงที่ขาดคำนวณตามลำดับจุด จึงไม่แม่นยำ",
                  "Làm trơn, đường nền và phép khớp qua khoảng thiếu tính theo chỉ số điểm nên kém chính xác.")),
    "spacing": dict(
        name=_t("点間隔の一様性", "Even spacing", "ความสม่ำเสมอของระยะห่าง", "Độ đều của khoảng cách"),
        criterion="Δxᵢ₊₁ / Δxᵢ ≤ {step_jump}",
        problem=_t("点間隔が急に変わる箇所があります（{where} cm⁻¹、最大比 {value}）。", "The point spacing jumps ({where} cm⁻¹; largest ratio {value}).",
                   "ระยะห่างระหว่างจุดเปลี่ยนกะทันหัน ({where} cm⁻¹ อัตราส่วนสูงสุด {value})", "Khoảng cách điểm thay đổi đột ngột ({where} cm⁻¹; tỉ số lớn nhất {value})."),
        advice=_t("つないだデータや編集したデータでないか確認してください。", "Check whether the data were joined or edited.",
                  "ตรวจว่าข้อมูลไม่ได้ถูกนำมาต่อหรือแก้ไข", "Kiểm tra xem dữ liệu có bị nối hoặc chỉnh sửa không.")),
    "snr": dict(
        name=_t("信号対雑音比", "Signal-to-noise ratio", "อัตราส่วนสัญญาณต่อสัญญาณรบกวน", "Tỉ số tín hiệu/nhiễu"),
        criterion="(max − median) / σ ≥ {snr}",
        info=_t("雑音 σ = {noise}（2 階差分から推定）。", "Noise σ = {noise} (from second differences).",
                "สัญญาณรบกวน σ = {noise} (จากผลต่างอันดับสอง)", "Nhiễu σ = {noise} (từ sai phân bậc hai)."),
        problem=_t("雑音（σ = {noise}）を上回る信号がほとんどありません。", "There is hardly any signal above the noise (σ = {noise}).",
                   "แทบไม่มีสัญญาณเหนือสัญญาณรบกวน (σ = {noise})", "Hầu như không có tín hiệu vượt nhiễu (σ = {noise})."),
        advice=_t("測定条件（露光・積算）を見直してください。", "Review the measurement (exposure, accumulations).",
                  "ทบทวนเงื่อนไขการวัด (เวลารับแสง จำนวนครั้งที่สะสม)", "Xem lại điều kiện đo (thời gian phơi, số lần tích lũy).")),
    "negative": dict(
        name=_t("負の強度", "Negative intensities", "ความเข้มติดลบ", "Cường độ âm"),
        criterion="y ≥ −3σ",
        problem=_t("雑音を超えて負の点が {value} 点あります（最小 {minimum}、{where} cm⁻¹）。", "{value} points are negative beyond the noise (lowest {minimum}; {where} cm⁻¹).",
                   "มี {value} จุดที่ติดลบเกินระดับสัญญาณรบกวน (ต่ำสุด {minimum}; {where} cm⁻¹)", "Có {value} điểm âm vượt mức nhiễu (thấp nhất {minimum}; {where} cm⁻¹)."),
        advice=_t("装置側でバックグラウンドを差し引いたデータか確認してください。", "Check whether the instrument already subtracted a background.",
                  "ตรวจว่าเครื่องมือได้ลบพื้นหลังไปแล้วหรือไม่", "Kiểm tra xem thiết bị đã trừ nền chưa.")),
    "saturation": dict(
        name=_t("頭打ち（飽和）", "Flat top (saturation)", "ยอดแบน (อิ่มตัว)", "Đỉnh phẳng (bão hòa)"),
        criterion=_t("最大値が 3 点以上続かない", "the maximum does not repeat on 3 or more points in a row",
                     "ค่าสูงสุดไม่ซ้ำติดกัน 3 จุดขึ้นไป", "giá trị lớn nhất không lặp 3 điểm liên tiếp trở lên"),
        problem=_t("最大値が {value} 点続いています（{where} cm⁻¹）。検出器の飽和が疑われます。", "The maximum repeats on {value} points in a row ({where} cm⁻¹); the detector may have saturated.",
                   "ค่าสูงสุดซ้ำติดกัน {value} จุด ({where} cm⁻¹) อาจเกิดการอิ่มตัวของตัวตรวจวัด", "Giá trị lớn nhất lặp {value} điểm liên tiếp ({where} cm⁻¹); có thể đầu dò bị bão hòa."),
        advice=_t("その帯の強度・形・面積は使わないでください。測定条件を変えて測り直すのが確実です。", "Do not use the intensity, shape or area of that band; measuring again with other settings is safest.",
                  "อย่าใช้ความเข้ม รูปร่าง หรือพื้นที่ของแถบนั้น การวัดใหม่ด้วยเงื่อนไขอื่นปลอดภัยที่สุด",
                  "Không dùng cường độ, hình dạng hay diện tích của dải đó; đo lại với thiết lập khác là chắc chắn nhất.")),
    "transmittance_over": dict(
        name=_t("透過率 100 % 超", "Transmittance above 100 %", "ค่าการส่งผ่านเกิน 100 %", "Độ truyền qua trên 100 %"),
        criterion="T ≤ 100 %",
        problem=_t("{value} 点が 100 % を超えています（{where} cm⁻¹）。", "{value} points exceed 100 % ({where} cm⁻¹).",
                   "{value} จุดเกิน 100 % ({where} cm⁻¹)", "{value} điểm vượt 100 % ({where} cm⁻¹)."),
        advice=_t("バックグラウンド（参照）測定の条件を確認してください。", "Check the background (reference) measurement.",
                  "ตรวจการวัดพื้นหลัง (อ้างอิง)", "Kiểm tra phép đo nền (tham chiếu).")),
    "absorbance_high": dict(
        name=_t("高い吸光度", "High absorbance", "ค่าการดูดกลืนแสงสูง", "Độ hấp thụ cao"),
        criterion="A ≤ {absorbance_high}",
        problem=_t("吸光度が {limit} を超える点が {value} 点あります（{where} cm⁻¹）。透過光がほとんどなく、帯の形・強度は信頼できません。",
                   "{value} points exceed absorbance {limit} ({where} cm⁻¹); almost no light is transmitted and band shapes and intensities are unreliable.",
                   "มี {value} จุดที่ค่าการดูดกลืนแสงเกิน {limit} ({where} cm⁻¹) แสงแทบไม่ผ่าน รูปร่างและความเข้มของแถบเชื่อถือไม่ได้",
                   "Có {value} điểm có độ hấp thụ vượt {limit} ({where} cm⁻¹); hầu như không có ánh sáng truyền qua, hình dạng và cường độ dải không đáng tin."),
        advice=_t("試料を薄くするか濃度を下げて測り直すことを検討してください。", "Consider a thinner or more dilute sample.",
                  "พิจารณาใช้ตัวอย่างที่บางลงหรือเจือจางลง", "Cân nhắc mẫu mỏng hơn hoặc loãng hơn.")),
    "points_used": dict(
        name=_t("表示範囲内の点数", "Points in the range", "จำนวนจุดในช่วง", "Số điểm trong khoảng"),
        criterion="≥ 5",
        problem=_t("表示範囲内の点が少なすぎます。", "Too few points in the range.", "จุดในช่วงน้อยเกินไป", "Quá ít điểm trong khoảng."),
        advice=_t("表示範囲を広げてください。", "Widen the range.", "ขยายช่วง", "Mở rộng khoảng.")),
    "spikes": dict(
        name=_t("スパイク除去", "Spike removal", "การลบสไปก์", "Loại gai nhiễu"),
        criterion=_t("置換点の隣の高さ比 < {spike_neighbour}", "neighbour height ratio of a replaced point < {spike_neighbour}",
                     "อัตราส่วนความสูงของจุดข้างเคียง < {spike_neighbour}", "tỉ số chiều cao điểm lân cận < {spike_neighbour}"),
        info=_t("{value} 点を置き換えました（{where} cm⁻¹）。", "{value} points were replaced ({where} cm⁻¹).",
                "แทนที่ {value} จุด ({where} cm⁻¹)", "Đã thay {value} điểm ({where} cm⁻¹)."),
        problem=_t("{value} 点を置き換え、そのうち {suspect_count} 点（{suspect} cm⁻¹）は隣の点も高く、本物の鋭いピークを削った可能性があります。",
                   "{value} points were replaced; {suspect_count} of them ({suspect} cm⁻¹) have raised neighbours and may be part of a real narrow peak.",
                   "แทนที่ {value} จุด ในจำนวนนี้ {suspect_count} จุด ({suspect} cm⁻¹) มีจุดข้างเคียงสูงด้วย อาจเป็นพีกแคบจริงที่ถูกตัด",
                   "Đã thay {value} điểm; {suspect_count} điểm ({suspect} cm⁻¹) có điểm lân cận cũng cao, có thể là đỉnh hẹp thật bị cắt."),
        advice=_t("「処理前のデータを薄く重ねる」で確かめ、本物のピークならスパイク除去を外すかしきい値を上げてください。",
                  "Check with “Overlay the data as measured”; if it is a real peak, turn spike removal off or raise its threshold.",
                  "ตรวจด้วย “ซ้อนข้อมูลก่อนประมวลผล” หากเป็นพีกจริงให้ปิดการลบสไปก์หรือเพิ่มเกณฑ์",
                  "Kiểm tra bằng “Chồng dữ liệu gốc”; nếu là đỉnh thật, hãy tắt loại gai nhiễu hoặc tăng ngưỡng.")),
    "normalization": dict(
        name=_t("正規化", "Normalisation", "การนอร์มัลไลซ์", "Chuẩn hóa"),
        criterion=_t("正規化できる", "normalisation applied", "นอร์มัลไลซ์ได้", "chuẩn hóa được"),
        problem=_t("正規化していません（{problem}：範囲 {window}）。このスペクトルは他と同じ尺度ではありません。",
                   "Not normalised ({problem}; window {window}). This spectrum is not on the same scale as the others.",
                   "ไม่ได้นอร์มัลไลซ์ ({problem}; ช่วง {window}) สเปกตรัมนี้ไม่อยู่ในสเกลเดียวกับสเปกตรัมอื่น",
                   "Chưa chuẩn hóa ({problem}; cửa sổ {window}). Phổ này không cùng thang với các phổ khác."),
        advice=_t("正規化の範囲を、このスペクトルのデータがある範囲にしてください。", "Choose a window inside this spectrum's data.",
                  "เลือกช่วงที่อยู่ภายในข้อมูลของสเปกตรัมนี้", "Chọn cửa sổ nằm trong dữ liệu của phổ này.")),
    "clipped": dict(
        name=_t("透過率 ≤ 0 の点", "Transmittance ≤ 0", "ค่าการส่งผ่าน ≤ 0", "Độ truyền qua ≤ 0"),
        criterion="0",
        problem=_t("{value} 点で透過率が 0 以下のため、吸光度 8 で打ち切りました（{where} cm⁻¹）。", "{value} points with transmittance at or below 0 were clipped to absorbance 8 ({where} cm⁻¹).",
                   "{value} จุดที่ค่าการส่งผ่าน ≤ 0 ถูกตัดที่ค่าการดูดกลืนแสง 8 ({where} cm⁻¹)", "{value} điểm có độ truyền qua ≤ 0 bị cắt ở độ hấp thụ 8 ({where} cm⁻¹)."),
        advice=_t("飽和した帯です。その帯の形・強度・フィットは使わないでください。", "These bands are saturated; do not use their shape, intensity or fits.",
                  "แถบเหล่านี้อิ่มตัว อย่าใช้รูปร่าง ความเข้ม หรือการฟิต", "Các dải này đã bão hòa; không dùng hình dạng, cường độ hay phép khớp của chúng.")),
    "peak_snr": dict(
        name=_t("ピークの S/N", "Peak S/N", "S/N ของพีก", "S/N của đỉnh"),
        criterion="prominence / σ ≥ {snr}",
        info=_t("検出したピークはすべて基準以上です。", "Every detected peak meets the criterion.", "พีกที่ตรวจพบทั้งหมดผ่านเกณฑ์", "Mọi đỉnh tìm được đều đạt tiêu chí."),
        problem=_t("{total} 個中 {value} 個のピークが雑音と区別できません（{where} cm⁻¹）。", "{value} of {total} peaks cannot be told from the noise ({where} cm⁻¹).",
                   "{value} จาก {total} พีกแยกจากสัญญาณรบกวนไม่ได้ ({where} cm⁻¹)", "{value} trên {total} đỉnh không phân biệt được với nhiễu ({where} cm⁻¹)."),
        advice=_t("ピーク一覧の S/N 列を見て、報告に使うか判断してください。感度を上げると減ります。", "See the S/N column of the peak table before reporting them; a higher sensitivity value lists fewer.",
                  "ดูคอลัมน์ S/N ในตารางพีกก่อนรายงาน การเพิ่มค่าความไวจะลดจำนวนลง",
                  "Xem cột S/N trong bảng đỉnh trước khi báo cáo; tăng độ nhạy sẽ liệt kê ít hơn.")),
    "peakfit_limit": dict(
        name=_t("単一ピークフィットの境界", "Single-peak fit bounds", "ขอบเขตการฟิตพีกเดี่ยว", "Giới hạn khớp một đỉnh"),
        criterion=_t("幅・中心が上限下限にない", "width and centre away from their bounds", "ความกว้างและจุดศูนย์กลางไม่อยู่ที่ขีดจำกัด", "độ rộng và tâm không nằm ở giới hạn"),
        problem=_t("{value} 個のフィットで幅か中心が上限・下限に到達しています（{where} cm⁻¹）。その半値幅・中心は決まっていません。",
                   "{value} fits ended with the width or centre on a bound ({where} cm⁻¹); their FWHM or centre is not determined.",
                   "{value} การฟิตมีความกว้างหรือจุดศูนย์กลางอยู่ที่ขีดจำกัด ({where} cm⁻¹) ค่า FWHM หรือจุดศูนย์กลางจึงยังไม่ถูกกำหนด",
                   "{value} phép khớp có độ rộng hoặc tâm nằm ở giới hạn ({where} cm⁻¹); FWHM hoặc tâm chưa được xác định."),
        advice=_t("フィットの半幅を見直してください。", "Review the fit half-window.", "ทบทวนครึ่งหน้าต่างสำหรับฟิต", "Xem lại nửa cửa sổ khớp.")),
    "peakfit_failed": dict(
        name=_t("単一ピークフィットの失敗", "Failed single-peak fits", "การฟิตพีกเดี่ยวที่ไม่สำเร็จ", "Khớp một đỉnh thất bại"),
        criterion="0",
        problem=_t("{value} 個のピークはフィットできず、表に値を入れていません（{where} cm⁻¹）。", "{value} peaks could not be fitted and have no values in the table ({where} cm⁻¹).",
                   "{value} พีกฟิตไม่ได้และไม่มีค่าในตาราง ({where} cm⁻¹)", "{value} đỉnh không khớp được và không có giá trị trong bảng ({where} cm⁻¹)."),
        advice=_t("重なったピークなら「ピーク分離フィット」を使ってください。", "For overlapping peaks use “Fit overlapping peaks”.",
                  "หากพีกซ้อนกันให้ใช้ “ฟิตแยกพีกที่ซ้อนกัน”", "Với các đỉnh chồng nhau hãy dùng “Khớp tách các đỉnh chồng nhau”.")),
    "fit_status": dict(
        name=_t("フィットの実行", "Fit", "การฟิต", "Phép khớp"),
        criterion=_t("収束し、結果が得られる", "converges and gives a result", "ลู่เข้าและได้ผลลัพธ์", "hội tụ và cho kết quả"),
        problem=_t("フィットできませんでした（{message}）。値は出力しません。", "The fit failed ({message}); no values are given.",
                   "ฟิตไม่สำเร็จ ({message}) ไม่แสดงค่า", "Phép khớp thất bại ({message}); không đưa ra giá trị."),
        advice=_t("範囲・初期位置・成分の数を見直してください。", "Review the range, starting positions and number of components.",
                  "ทบทวนช่วง ตำแหน่งเริ่มต้น และจำนวนองค์ประกอบ", "Xem lại khoảng, vị trí ban đầu và số thành phần.")),
    "dof": dict(
        name=_t("データ点数 / パラメータ数", "Points / parameters", "จำนวนจุด / พารามิเตอร์", "Số điểm / tham số"),
        criterion=_t("点数 > パラメータ数", "points > parameters", "จำนวนจุด > พารามิเตอร์", "số điểm > số tham số"),
        info=_t("{points} 点に {parameters} 個のパラメータ。", "{parameters} parameters on {points} points.",
                "{parameters} พารามิเตอร์บน {points} จุด", "{parameters} tham số trên {points} điểm.")),
    "chi2": dict(
        name=_t("換算 χ²", "Reduced χ²", "Reduced χ²", "χ² rút gọn"),
        criterion="≤ {chi2_caution}: PASS · ≤ {chi2_warning}: CAUTION",
        problem=_t("残差が雑音より大きく、モデルが説明していない構造があります。", "The residual is larger than the noise: the model leaves structure unexplained.",
                   "ส่วนเหลือมากกว่าสัญญาณรบกวน แบบจำลองยังอธิบายโครงสร้างบางส่วนไม่ได้",
                   "Phần dư lớn hơn nhiễu: mô hình còn bỏ sót cấu trúc."),
        advice=_t("残差のグラフを見て、成分の数・関数・バックグラウンドを見直してください。", "Look at the residual plot and review the number of components, the model and the background.",
                  "ดูกราฟส่วนเหลือ แล้วทบทวนจำนวนองค์ประกอบ ฟังก์ชัน และพื้นหลัง", "Xem đồ thị phần dư và xem lại số thành phần, hàm và nền.")),
    "autocorr": dict(
        name=_t("残差の自己相関", "Residual autocorrelation", "สหสัมพันธ์ในตัวของส่วนเหลือ", "Tự tương quan phần dư"),
        criterion="≤ {autocorr}",
        problem=_t("残差が波打っています（取りこぼした成分か、形の合わない関数）。", "The residual has waves (a missing component or a mismatched shape).",
                   "ส่วนเหลือเป็นคลื่น (ขาดองค์ประกอบหรือรูปร่างฟังก์ชันไม่ตรง)", "Phần dư có dạng sóng (thiếu thành phần hoặc dạng hàm không khớp)."),
        advice=_t("残差のグラフで波の位置を確かめてください。", "Find where the waves are on the residual plot.", "ดูตำแหน่งของคลื่นในกราฟส่วนเหลือ", "Tìm vị trí sóng trên đồ thị phần dư.")),
    "correlation": dict(
        name=_t("最大のパラメータ相関", "Largest parameter correlation", "สหสัมพันธ์ของพารามิเตอร์สูงสุด", "Tương quan tham số lớn nhất"),
        criterion="|r| < {correlation}",
        problem=_t("パラメータ同士が打ち消し合えるため、個々の値はデータから決まっていません。", "Two parameters trade off, so their individual values are not determined by the data.",
                   "พารามิเตอร์ชดเชยกันได้ ค่าแต่ละตัวจึงไม่ถูกกำหนดโดยข้อมูล", "Hai tham số bù trừ nhau nên giá trị riêng không được xác định bởi dữ liệu."),
        advice=_t("成分を減らすか、範囲を広げてください。", "Use fewer components or a wider range.", "ลดจำนวนองค์ประกอบหรือขยายช่วง", "Dùng ít thành phần hơn hoặc mở rộng khoảng.")),
    "no_errors": dict(
        name=_t("誤差の見積もり", "Error estimate", "การประมาณค่าคลาดเคลื่อน", "Ước lượng sai số"),
        criterion=_t("共分散行列が求まる", "covariance matrix obtained", "ได้เมทริกซ์ความแปรปรวนร่วม", "có ma trận hiệp phương sai"),
        problem=_t("誤差を見積もれません。パラメータがデータから決まっていません。", "No errors could be estimated: the parameters are not determined by the data.",
                   "ประมาณค่าคลาดเคลื่อนไม่ได้ พารามิเตอร์ไม่ถูกกำหนดโดยข้อมูล", "Không ước lượng được sai số: các tham số không được xác định bởi dữ liệu."),
        advice=_t("成分を減らしてください。", "Use fewer components.", "ลดจำนวนองค์ประกอบ", "Dùng ít thành phần hơn.")),
    "comp_snr": dict(
        name=_t("成分の高さ / 雑音", "Component height / noise", "ความสูงองค์ประกอบ / สัญญาณรบกวน", "Chiều cao thành phần / nhiễu"),
        criterion="≥ {snr}",
        problem=_t("この成分は雑音の {snr} 倍に満たず、有意とは言えません。", "This component is lower than {snr} × the noise and is not significant.",
                   "องค์ประกอบนี้ต่ำกว่า {snr} เท่าของสัญญาณรบกวน จึงไม่มีนัยสำคัญ", "Thành phần này thấp hơn {snr} lần nhiễu nên không có ý nghĩa."),
        advice=_t("成分を外してフィットし直すことを検討してください。", "Consider fitting without this component.", "พิจารณาฟิตใหม่โดยไม่มีองค์ประกอบนี้", "Cân nhắc khớp lại không có thành phần này.")),
    "comp_limit": dict(
        name=_t("上限・下限への到達", "Parameters on a bound", "พารามิเตอร์ที่ขีดจำกัด", "Tham số ở giới hạn"),
        criterion=_t("どのパラメータも境界にない", "no parameter on a bound", "ไม่มีพารามิเตอร์อยู่ที่ขีดจำกัด", "không tham số nào ở giới hạn"),
        problem=_t("{parameters} が上限・下限に到達しています。その値は拘束で決まっており、誤差は示しません。",
                   "{parameters} ended on a bound; the value is set by the constraint and no error is given.",
                   "{parameters} อยู่ที่ขีดจำกัด ค่าถูกกำหนดโดยข้อจำกัด จึงไม่แสดงค่าคลาดเคลื่อน",
                   "{parameters} nằm ở giới hạn; giá trị do ràng buộc quyết định và không đưa ra sai số."),
        advice=_t("初期位置・中心の動ける幅・範囲を見直してください。", "Review the starting position, the allowed shift and the range.",
                  "ทบทวนตำแหน่งเริ่มต้น ระยะเลื่อนที่อนุญาต และช่วง", "Xem lại vị trí ban đầu, độ dịch cho phép và khoảng.")),
    "area_err": dict(
        name=_t("面積の相対誤差", "Relative error of the area", "ค่าคลาดเคลื่อนสัมพัทธ์ของพื้นที่", "Sai số tương đối của diện tích"),
        criterion="≤ {area_rel_err_pct} %",
        problem=_t("面積の不確かさが大きく、面積比の比較には向きません。", "The area is too uncertain for comparing area ratios.",
                   "พื้นที่ไม่แน่นอนมากเกินไปสำหรับการเปรียบเทียบอัตราส่วน", "Diện tích quá bất định để so sánh tỉ số."),
        advice=_t("雑音を減らす（積算）か、範囲を見直してください。", "Reduce the noise (more accumulations) or review the range.",
                  "ลดสัญญาณรบกวน (สะสมมากขึ้น) หรือทบทวนช่วง", "Giảm nhiễu (tích lũy thêm) hoặc xem lại khoảng.")),
    "narrow": dict(
        name=_t("成分の幅（点数）", "Component width (points)", "ความกว้างองค์ประกอบ (จุด)", "Độ rộng thành phần (điểm)"),
        criterion="FWHM ≥ 4 Δx",
        problem=_t("半値幅がデータ点 4 つ分より狭く、スパイクや雑音に合わせている可能性があります。", "The FWHM is narrower than four points; the component may follow a spike or noise.",
                   "FWHM แคบกว่าสี่จุด องค์ประกอบอาจกำลังตามสไปก์หรือสัญญาณรบกวน", "FWHM hẹp hơn bốn điểm; thành phần có thể đang bám theo gai nhiễu hoặc nhiễu."),
        advice=_t("処理前のデータと重ねて確かめてください。", "Compare with the data as measured.", "เปรียบเทียบกับข้อมูลก่อนประมวลผล", "So sánh với dữ liệu gốc.")),
    "r2": dict(name=_t("R²（参考）", "R² (reference)", "R² (อ้างอิง)", "R² (tham khảo)"), criterion="—"),
    "rmse": dict(name=_t("RMSE（参考）", "RMSE (reference)", "RMSE (อ้างอิง)", "RMSE (tham khảo)"), criterion="—"),
}

NORMALIZE_PROBLEMS = {
    "window_empty": _t("範囲に点がない", "no points in the window", "ไม่มีจุดในช่วง", "không có điểm trong cửa sổ"),
    "flat": _t("スペクトルが平坦", "the spectrum is flat", "สเปกตรัมแบนราบ", "phổ phẳng"),
}

FIT_MESSAGES = {
    "too_few_points": _t("範囲内の点が足りません", "too few points in the range", "จำนวนจุดในช่วงน้อยเกินไป", "quá ít điểm trong khoảng"),
    "center_outside": _t("初期位置がデータの範囲外です", "a starting position lies outside the data", "ตำแหน่งเริ่มต้นอยู่นอกช่วงข้อมูล", "một vị trí ban đầu nằm ngoài dữ liệu"),
    "flat": _t("範囲内のデータが平坦です", "the data are flat in the range", "ข้อมูลในช่วงนี้แบนราบ", "dữ liệu trong khoảng này phẳng"),
    "too_many": _t("成分が多すぎます", "too many components", "องค์ประกอบมากเกินไป", "quá nhiều thành phần"),
    "no_centers": _t("初期位置がありません", "no starting positions", "ไม่มีตำแหน่งเริ่มต้น", "không có vị trí ban đầu"),
    "": _t("収束しませんでした", "no convergence", "ไม่ลู่เข้า", "không hội tụ"),
}


def _pick(entry, language: str) -> str:
    if isinstance(entry, dict):
        return entry.get(language) or entry["en"]
    return entry or ""


class _Defaults(dict):
    def __missing__(self, key):
        return "—"


def render(it: dict, language: str = "en", thresholds: dict | None = None) -> dict:
    """name, criterion, reason and advice of a QC item in ``language`` ('ja', 'en', 'th', 'vi')."""
    language = language if language in ("ja", "en", "th", "vi") else "en"
    entry = CHECKS.get(it["code"], {})
    params = _Defaults({k: v for k, v in (thresholds or {}).items()})
    if thresholds and "area_rel_err" in thresholds:
        params["area_rel_err_pct"] = f"{100 * thresholds['area_rel_err']:g}"
    for key, value in (thresholds or {}).items():
        params[key] = f"{value:g}" if isinstance(value, (int, float)) and math.isfinite(value) else value
    params.update({k: v for k, v in it.get("params", {}).items()})
    params["value"] = it.get("value") if it.get("value") is not None else "—"
    if it["code"] == "normalization" and it["params"].get("problem"):
        params["problem"] = _pick(NORMALIZE_PROBLEMS.get(it["params"]["problem"], it["params"]["problem"]), language)
    if it["code"] == "fit_status":
        message = str(it["params"].get("message", ""))
        key = message if message in FIT_MESSAGES else ""
        params["message"] = _pick(FIT_MESSAGES[key], language) + ("" if key or not message else f" — {message[:120]}")
    status = it["status"]
    if status == "PASS":
        reason = _pick(entry.get("info"), language) or _pick(PASS_REASON, language)
        advice = ""
    elif status == "NOT EVALUATED":
        reason = _pick(NOT_EVALUATED_REASONS.get(it["params"].get("reason", ""), NOT_EVALUATED_REASONS[""]), language)
        advice = ""
    else:
        reason = _pick(entry.get("problem"), language)
        advice = _pick(entry.get("advice"), language)
    subject = it["subject"] + (f" · #{it['component']}" if it.get("component") else "")
    return {
        "section": _pick(SECTIONS.get(it["section"], it["section"]), language),
        "subject": subject,
        "name": _pick(entry.get("name", it["code"]), language),
        "criterion": _pick(entry.get("criterion", "—"), language).format_map(params),
        "reason": reason.format_map(params),
        "advice": advice.format_map(params),
    }


# ------------------------------------------------------------------------------- the QC panel
STATUS_BADGES = {"PASS": "✅ PASS", "CAUTION": "🟡 CAUTION", "WARNING": "🟠 WARNING", "FAIL": "🔴 FAIL",
                 "NOT EVALUATED": "⚪ NOT EVALUATED"}

PANEL = {
    "title": _t("🔎 QC Report（解析の品質）", "🔎 QC Report (analysis quality)", "🔎 QC Report (คุณภาพการวิเคราะห์)", "🔎 QC Report (chất lượng phân tích)"),
    "alert": _t("QC: {status}（{count} 件）。数値を使う前に QC Report を確認してください。", "QC: {status} ({count} items). Check the QC Report before using the numbers.",
                "QC: {status} ({count} รายการ) ตรวจ QC Report ก่อนใช้ตัวเลข", "QC: {status} ({count} mục). Hãy xem QC Report trước khi dùng số liệu."),
    "about": _t("表示中のスペクトルについて、入力データ・前処理・ピーク・フィットを自動で点検します。総合評価は最も悪い項目で決まり、他の項目で相殺されません。"
                "プログラムが最後まで動いたことと、結果が科学的に妥当なことは別です。この点検は下の基準の範囲でしか判断せず、帰属やベースラインの物理的な妥当性は判断しません。",
                "Checks the input, processing, peaks and fits of the visible spectra. The overall status is the worst item and is never offset by items that pass. "
                "A run without errors is not the same as a sound result; these checks judge only the criteria below, not assignments or whether a baseline is physically right.",
                "ตรวจข้อมูลนำเข้า การประมวลผล พีก และการฟิตของสเปกตรัมที่แสดงอยู่ ผลรวมคือรายการที่แย่ที่สุด และไม่ถูกชดเชยด้วยรายการที่ผ่าน "
                "การที่โปรแกรมทำงานจบไม่ได้แปลว่าผลถูกต้องทางวิทยาศาสตร์ การตรวจนี้ตัดสินเฉพาะเกณฑ์ด้านล่าง ไม่ตัดสินการระบุพีกหรือความถูกต้องทางกายภาพของเบสไลน์",
                "Kiểm tra dữ liệu đầu vào, xử lý, đỉnh và phép khớp của các phổ đang hiện. Kết quả chung là mục xấu nhất và không bị bù trừ bởi các mục đạt. "
                "Chương trình chạy xong không có nghĩa kết quả đúng về khoa học; các kiểm tra này chỉ xét các tiêu chí dưới đây, không xét việc gán đỉnh hay tính đúng vật lý của đường nền."),
    "overall": _t("総合評価", "Overall", "ผลรวม", "Tổng thể"),
    "show_all": _t("PASS の項目も表示", "Show items that pass", "แสดงรายการที่ผ่านด้วย", "Hiện cả các mục đạt"),
    "nothing": _t("表示する項目はありません（すべて PASS）。", "Nothing to show (everything passes).", "ไม่มีรายการที่ต้องแสดง (ผ่านทั้งหมด)", "Không có mục nào để hiện (tất cả đều đạt)."),
    "col_status": _t("判定", "Status", "ผล", "Kết quả"),
    "col_section": _t("区分", "Section", "หมวด", "Mục"),
    "col_subject": _t("スペクトル", "Spectrum", "สเปกตรัม", "Phổ"),
    "col_name": _t("項目", "Check", "รายการ", "Kiểm tra"),
    "col_value": _t("値", "Value", "ค่า", "Giá trị"),
    "col_criterion": _t("基準", "Criterion", "เกณฑ์", "Tiêu chí"),
    "col_reason": _t("理由", "Reason", "เหตุผล", "Lý do"),
    "col_advice": _t("確認すること", "What to check", "สิ่งที่ควรตรวจ", "Cần kiểm tra"),
    "processing_log": _t("実施した処理", "Processing applied", "การประมวลผลที่ทำ", "Xử lý đã áp dụng"),
    "not_evaluated": _t("未評価の項目", "Not evaluated", "รายการที่ไม่ได้ประเมิน", "Mục chưa đánh giá"),
    "thresholds": _t("判定基準（暫定値・変更できます）", "Criteria (provisional, editable)", "เกณฑ์ (ค่าชั่วคราว แก้ไขได้)", "Tiêu chí (tạm thời, có thể chỉnh)"),
    "thresholds_note": _t("どの値も一般的な目安による暫定値です。分野や装置で妥当な値が違う場合は変更してください（作業ファイルに保存され、QC レポートに記録されます）。",
                          "Every value is a provisional rule of thumb. Change it where your field or instrument calls for another value (it is saved in the project and recorded in the QC report).",
                          "ทุกค่าเป็นค่าประมาณชั่วคราว เปลี่ยนได้หากสาขาหรือเครื่องมือของคุณต้องการค่าอื่น (บันทึกในโปรเจกต์และในรายงาน QC)",
                          "Mọi giá trị là quy tắc kinh nghiệm tạm thời. Hãy đổi nếu lĩnh vực hoặc thiết bị của bạn cần giá trị khác (được lưu trong dự án và ghi vào báo cáo QC)."),
    "reset": _t("既定値に戻す", "Reset to defaults", "คืนค่าเริ่มต้น", "Đặt lại mặc định"),
    "download_json": _t("📥 QC レポート（JSON：条件・履歴・結果）", "📥 QC report (JSON: settings, history, results)",
                        "📥 รายงาน QC (JSON: การตั้งค่า ประวัติ ผลลัพธ์)", "📥 Báo cáo QC (JSON: thiết lập, lịch sử, kết quả)"),
    "download_csv": _t("📥 QC 項目（CSV）", "📥 QC items (CSV)", "📥 รายการ QC (CSV)", "📥 Các mục QC (CSV)"),
    "files_note": _t("CSV の列名は言語によらず英語です。QC が PASS でない CSV には、先頭に # で始まる注意行が入り、WARNING・FAIL のときはファイル名に _QC-WARNING などが付きます。",
                     "CSV column names are in English whatever the interface language. CSV files from results that do not pass start with lines beginning with #, and WARNING or FAIL adds _QC-WARNING etc. to the file name.",
                     "ชื่อคอลัมน์ใน CSV เป็นภาษาอังกฤษเสมอ CSV ของผลที่ไม่ผ่านจะขึ้นต้นด้วยบรรทัดที่เริ่มด้วย # และกรณี WARNING หรือ FAIL ชื่อไฟล์จะมี _QC-WARNING เป็นต้น",
                     "Tên cột CSV luôn bằng tiếng Anh. Tệp CSV của kết quả không đạt bắt đầu bằng các dòng có #, và với WARNING hoặc FAIL tên tệp có thêm _QC-WARNING v.v."),
    "fit_summary": _t("このフィットの QC: {status}。理由は QC Report を見てください。", "QC of this fit: {status}. See the QC Report for the reasons.",
                      "QC ของการฟิตนี้: {status} ดูเหตุผลใน QC Report", "QC của phép khớp này: {status}. Xem lý do trong QC Report."),
    "log_range": _t("表示範囲で切り出し：{low}–{high} cm⁻¹", "Cropped to {low}–{high} cm⁻¹", "ตัดช่วง {low}–{high} cm⁻¹", "Cắt khoảng {low}–{high} cm⁻¹"),
    "log_ftir": _t("FT-IR：吸光度に変換して処理（A = −log₁₀T、T ≤ 0 は A = 8 で打ち切り）、表示は {display}",
                   "FT-IR: processed as absorbance (A = −log₁₀T; T ≤ 0 clipped to A = 8), shown as {display}",
                   "FT-IR: ประมวลผลเป็นค่าการดูดกลืนแสง (A = −log₁₀T; T ≤ 0 ตัดที่ A = 8) แสดงเป็น {display}",
                   "FT-IR: xử lý theo độ hấp thụ (A = −log₁₀T; T ≤ 0 cắt ở A = 8), hiển thị dạng {display}"),
    "log_spikes": _t("スパイク除去（Whitaker–Hayes、しきい値 {threshold}）：{count} 点を置換", "Spike removal (Whitaker–Hayes, threshold {threshold}): {count} points replaced",
                     "ลบสไปก์ (Whitaker–Hayes เกณฑ์ {threshold}): แทนที่ {count} จุด", "Loại gai nhiễu (Whitaker–Hayes, ngưỡng {threshold}): thay {count} điểm"),
    "log_smooth": _t("平滑化（Savitzky–Golay、窓 {window} 点、次数 {order}）", "Smoothing (Savitzky–Golay, window {window} points, order {order})",
                     "ทำให้เรียบ (Savitzky–Golay หน้าต่าง {window} จุด อันดับ {order})", "Làm trơn (Savitzky–Golay, cửa sổ {window} điểm, bậc {order})"),
    "log_baseline": _t("ベースライン差し引き：{method} {params}", "Baseline subtracted: {method} {params}", "ลบเบสไลน์: {method} {params}", "Trừ đường nền: {method} {params}"),
    "log_normalize": _t("正規化：{mode} {window}", "Normalisation: {mode} {window}", "นอร์มัลไลซ์: {mode} {window}", "Chuẩn hóa: {mode} {window}"),
    "log_none": _t("前処理なし（読み込んだ値のまま）", "No processing (values as read)", "ไม่มีการประมวลผล (ค่าตามที่อ่าน)", "Không xử lý (giá trị như khi đọc)"),
    "log_note": _t("元データ（読み込んだファイル）は変更していません。処理はすべて設定どおりで、自動で切り替える処理はありません。",
                   "The data as read are never changed. Every step follows the settings; nothing is switched automatically.",
                   "ไม่มีการแก้ไขข้อมูลที่อ่านเข้ามา ทุกขั้นตอนเป็นไปตามการตั้งค่า ไม่มีการสลับวิธีอัตโนมัติ",
                   "Dữ liệu đã đọc không bao giờ bị thay đổi. Mọi bước theo thiết lập; không có gì tự động chuyển đổi."),
}

THRESHOLD_LABELS = {
    "snr": _t("S/N の下限", "Lowest S/N", "S/N ต่ำสุด", "S/N thấp nhất"),
    "chi2_caution": _t("換算 χ²：CAUTION の境", "Reduced χ²: CAUTION above", "Reduced χ²: CAUTION เมื่อเกิน", "χ² rút gọn: CAUTION khi vượt"),
    "chi2_warning": _t("換算 χ²：WARNING の境", "Reduced χ²: WARNING above", "Reduced χ²: WARNING เมื่อเกิน", "χ² rút gọn: WARNING khi vượt"),
    "autocorr": _t("残差の自己相関の上限", "Highest residual autocorrelation", "สหสัมพันธ์ในตัวของส่วนเหลือสูงสุด", "Tự tương quan phần dư cao nhất"),
    "correlation": _t("パラメータ相関の上限", "Highest parameter correlation", "สหสัมพันธ์ของพารามิเตอร์สูงสุด", "Tương quan tham số cao nhất"),
    "area_rel_err": _t("面積の相対誤差の上限", "Highest relative area error", "ค่าคลาดเคลื่อนสัมพัทธ์ของพื้นที่สูงสุด", "Sai số tương đối diện tích cao nhất"),
    "spike_neighbour": _t("スパイク：隣の点の高さ比の上限", "Spikes: highest neighbour ratio", "สไปก์: อัตราส่วนจุดข้างเคียงสูงสุด", "Gai nhiễu: tỉ số điểm lân cận cao nhất"),
    "step_jump": _t("隣り合う点間隔の比の上限", "Highest ratio of adjacent spacings", "อัตราส่วนระยะห่างที่ติดกันสูงสุด", "Tỉ số khoảng cách kề nhau cao nhất"),
    "gap": _t("抜けとみなす間隔（中央値の倍）", "Spacing taken as a gap (× median)", "ระยะห่างที่ถือว่าขาด (× มัธยฐาน)", "Khoảng cách coi là thiếu (× trung vị)"),
    "absorbance_high": _t("吸光度の上限", "Highest absorbance", "ค่าการดูดกลืนแสงสูงสุด", "Độ hấp thụ cao nhất"),
}


def panel(key: str, language: str, **values) -> str:
    text = _pick(PANEL[key], language)
    return text.format(**values) if values else text
