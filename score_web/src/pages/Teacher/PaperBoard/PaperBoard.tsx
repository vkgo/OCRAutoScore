import React, {useEffect, useState} from "react"
import {Button } from 'antd'
import {PlusCircleOutlined} from '@ant-design/icons'
import './PaperBoard.less'
import PaperList from "@/components/PaperList/PaperList"
import { useHistory } from 'react-router-dom';
import axios from "axios"
const PaperBoard : React.FC = () => {
    const history = useHistory()
    const [papers, setPapers] = useState([])

    useEffect(()=>{
        getPaperList()
    }, [])

    const getPaperList = async () => {
        const result = await axios.request({
            url: 'teacher/papers',
            method: 'GET',
            params: {"username": window.sessionStorage.getItem('username')}
        })
        if(result.data.msg === 'success') {
            setPapers(result.data.papers)
        }
    }

    const deletePaper = async(paperId: number): Promise<void> => {
        await axios.request({
            method: "GET",
            url: "paper/delete",
            params: {"paperId":paperId}
        })
        getPaperList()
    }

    return (
        <div className="teacher_PaperBoard">
            <h2>我发布过的试卷</h2>
            <PaperList baseUrl="/teacher/list/detail/" list={papers} buttonText="查看学生作答情况" showDeleteButton={true} deleteFunction={deletePaper}/> 
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