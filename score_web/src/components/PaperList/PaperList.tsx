import PaperListItem from "@/ts/interface/paperListItem_interface";
import React from "react";
import {List, Button, Space } from 'antd'
import './PaperList.less'
import { useHistory } from 'react-router-dom';
interface PaperListProps {
    baseUrl:string,
    list?: PaperListItem[],
    buttonText:string,
    showDeleteButton:boolean,
    deleteFunction?: (id:number) => void
}
const PaperList: React.FC<PaperListProps> = (props) => {
    const history = useHistory()
    const descriptionHtml = (item: PaperListItem) => {
        return (
            <div className="paper_list_description_container">
                {
                    item.teacher ? (
                        <div className="tag_container">
                            <img src={require("@/assets/user.png")} alt="教师图标"/>
                            <span>出卷老师: {item.teacher}</span>
                        </div>
                    ) : ''
                }
                { item.hot ? (
                    <div className="tag_container">
                        <img src={require("@/assets/hot.png")} alt="热度图标"/>
                        <span>共有{item.hot}学生做过</span>
                    </div>) : ''
                }
                {  item.avarageScore ? (
                    <div className="tag_container">
                        <img src={require("@/assets/averageScore.png")} alt="平均分图标"/>
                        <span>平均分 {item.avarageScore}分</span>
                    </div>): null
                }
                <div className="tag_container">
                    <img src={require("@/assets/time.png")} alt="上传时间图标"/>
                    <span>试卷发布时间 {item.time}</span>
                </div>
            </div>
        )   
    }

    return (
        <List
            dataSource={props.list}
            size="large"
            renderItem={(item: PaperListItem) =>{
                return (
                    <>
                        {
                            item.title !== '' ?
                                (<List.Item key={item.id}>
                                    <List.Item.Meta
                                        avatar={<img src={require("@/assets/paper.png")} alt="试卷图标"/>}
                                        title={<span>{item.title}</span>}
                                        description={descriptionHtml(item)}
                                    />
                                    <Space>
                                        <Button type="primary" onClick={()=>{history.push(props.baseUrl+item.id)}}>{props.buttonText}</Button>
                                        {props.showDeleteButton ? <Button type="primary" danger onClick={()=>props.deleteFunction(item.id)}>删除试卷</Button>:''}
                                    </Space>
                                </List.Item>
                                ): ''
                        }
                    </>
                )
            }}    
        />
    )
}

export default PaperList;