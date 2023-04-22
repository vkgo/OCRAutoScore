export default interface PaperListItem {
    id: number;
    title: string;
    time: string;
    hot?: number; // 多少学生做过这份试卷
    avarageScore?: number; // 平均分
    teacher?: string; // 发布试卷的老师
}