import React from "react"
import {Button } from 'antd'
import {PlusCircleOutlined} from '@ant-design/icons'
import './PaperBoard.less'
import PaperList from "@/components/PaperList/PaperList"
import { useHistory } from 'react-router-dom';
const PaperBoard : React.FC = () => {
    const history = useHistory()
    return (
        <div>
            <h2>我发布过的试卷</h2>
            <PaperList/> 
            <div className="button_container">
                <Button 
                    type="primary" 
                    block icon={<PlusCircleOutlined/>} 
                    style={{width:'15%'}} 
                    onClick={()=>{history.push("/teacher/list/add")}}>
                         添加新试卷
                </Button> 
            </div>
        </div>
    )
}

export default PaperBoard