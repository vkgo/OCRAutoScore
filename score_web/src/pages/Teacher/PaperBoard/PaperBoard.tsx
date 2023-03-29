import React from "react"
import {List, Button } from 'antd'
import {PlusCircleOutlined} from '@ant-design/icons'
import './PaperBoard.less'
interface PaperItemDataSource {
    id: number;
    title: string;
    time: string;
    hot: number; // 多少学生做过这份试卷
    avarageScore?: number; // 平均分
}
const PaperBoard : React.FC = () => {
    let dataSource: PaperItemDataSource[] = [
        {
            id: 0,
            title: "河北省石家庄市赵县2022-2023学年七年级下学期3月月考试题",
            time:'2011-10-31',
            hot: 2,
            avarageScore: 80
        },
        {
            id: 0,
            title: "浙江省嘉兴市平湖市2022-2023学年八年级上学期期末试题",
            time:'2012-12-01',
            hot: 10,
            avarageScore:78
        },
        {
            id: 0,
            title: "黑龙江省哈尔滨市香坊区风华中学2022-2023学年九年级3月份月考试题",
            time:'2023-02-15',
            hot: 0
        },
    ]

    const descriptionHtml = (item: PaperItemDataSource) => {
        return (
            <div className="description_container">
                <div className="tag_container">
                    <img src={require("@/assets/hot.png")} alt="热度图标"/>
                    <span>共有{item.hot}学生做过</span>
                </div>
                {  item.avarageScore ? (
                    <div className="tag_container">
                        <img src={require("@/assets/averageScore.png")} alt="平均分图标"/>
                        <span>平均分 {item.avarageScore}分</span>
                    </div>): null
                }
                <div className="tag_container">
                    <img src={require("@/assets/time.png")} alt="上传时间图标"/>
                    <span>{item.time}</span>
                </div>
            </div>
        )   
    }
    return (
        <div>
            <h2>我的试卷</h2>
            <List
                dataSource={dataSource}
                size="large"
                renderItem={(item: PaperItemDataSource)=>(
                    <List.Item key={item.id}>
                        <List.Item.Meta
                            avatar={<img src={require("@/assets/paper.png")} alt="试卷图标"/>}
                            title={<span>{item.title}</span>}
                            description={descriptionHtml(item)}
                        />
                        <Button type="primary">点击查看</Button>
                    </List.Item>
                )}    
            />
            <div className="button_container">
                <Button type="primary" block icon={<PlusCircleOutlined/>} style={{width:'15%'}}>添加新试卷</Button> 
            </div>
        </div>
    )
}

export default PaperBoard