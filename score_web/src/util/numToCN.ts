/**根据数字获取汉字*/
export default function numToCN(num: number): string {
    let words = ["零", "一", "二", "三", "四", "五", "六", "七", "八", "九", "十"];
    let adds = ["", '十', '百', '千', '万', '亿', '十', '百', '千'];
    if (words[num]) {
        return words[num];
    }
    else if (num > 10 && num < 20) {
        let numStr = num.toString();
        let n = numStr.substring(1, 2);
        let result = adds[1] + words[n];
        return result;
    }
    else if (num > 10) {
        let result = "";
        let numStr = num.toString();
        for (var i = 0; i < numStr.length; ++i) {
            let n = numStr.substring(i, i + 1);
            let m = numStr.length - i - 1;
            result += words[n] + adds[m];
        }
        return result;
    }
    else return "零";
} 