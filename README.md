# 以父母教育程度分析對小孩的影響

## 一、摘要

使用 report.py 整理 data 內資料集，觀察各項指標對學生成績是否有影響。其中，我們猜測「家長或監護人的學歷」對學生表現影響最大，故以觀察「家長或監護人的學歷」是否會影響成績為主要目的。

可能影響學生成績因素包含以下五點：
1. 家長或監護人的學歷
2. 考前是否複習
3. 學生性別
4. 種族
5. 午餐狀況

## 二、檔案說明

1. 資料集：
    - data 內存放有「學生成績與父母教育程度」的 exams.csv 檔案
    - 資料來源：[Exam Scores](<http://roycekimmons.com/tools/generated_data/exams> "Exam Scores 資料來源")
2. 程式碼：
    report.py
    - 機器學習方法
        Random Forest Classifier 隨機林分類器
    - 模型評估
        - scikit-learn 的 classification_report：顯示分類模型的性能指標。用於評估分類問題的預測結果。
        - randomforest.score：可用於計算模型在測試數據集上的分類準確率。
        - randomforest.feature_importances_：可獲得每個對於預測目標特徵的重要性分數。
        - Silhouette coefficient：可用來評估分類結果的好壞。

3. 結論與詳細說明：
    請閱讀「[以父母教育程度分析對小孩成績的影響](<https://github.com/WenHsin0130/exams/blob/main/%E4%BB%A5%E7%88%B6%E6%AF%8D%E6%95%99%E8%82%B2%E7%A8%8B%E5%BA%A6%E5%88%86%E6%9E%90%E5%B0%8D%E5%B0%8F%E5%AD%A9%E6%88%90%E7%B8%BE%E7%9A%84%E5%BD%B1%E9%9F%BF.pdf> "以父母教育程度分析對小孩成績的影響.pdf")」