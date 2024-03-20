import React,{useState, useEffect} from 'react';
import { Button, Image, Popconfirm, message } from 'antd'
import { useParams } from "react-router-dom";
import './PaperDetail.less'
import axios from 'axios';
import ImageUpload from '@/components/ImageUpload/ImageUpload';
import type { UploadFile } from 'antd/es/upload/interface';

const PaperDetail: React.FC = () => {
    const paperId =  parseInt(useParams<{id: string}>()["id"])
    const [paperImages, setImageList] = useState([])
    const [open, setOpen] = useState(false);
    const [messageApi, contextHolder] = message.useMessage()
    const [score, setScore] = useState(-1);
    const [answers, setAnswers] = useState<UploadFile[]>([]);

    const getPaperPhotos = async () => {
        const result = await axios.request({
            url:"student/paper/detail",
            method: "GET",
            params: {paperId}
        })
        if(result.data.msg === 'success') {
            setImageList(result.data.paperImages)
        }
    }

    const getPaperAnswersBefore = async () => {
        const result = await axios.request({
            url: "student/paper/answer/detail",
            method: "GET",
            params: {
                paperId: paperId,
                username: window.sessionStorage.getItem('username')
            }
        })
        if(result.data.msg === 'success') {
            setAnswers(result.data.answerImages)
        }
    }

    useEffect(()=>{
        if(paperImages.length ===  0) getPaperPhotos()
        if(answers.length === 0) getPaperAnswersBefore()
    })

    const waitingForScore = async() => {
        setOpen(false);
        messageApi.open({
            type: 'loading',
            content: '正在评分中',
            duration: 0
        })

        const result = await axios.request({
            url: 'student/paper/score',
            method: 'GET',
            params: {
                paperId: paperId,
                username: window.sessionStorage.getItem('username')
            }
        })

        console.log(result)
        
        setTimeout(()=>{messageApi.destroy();setScore(80);}, 3000)
    }

    const handleAnswerChange = (newFiles: UploadFile[]): void => {
       setAnswers(newFiles);
    };

    const handlePhotoRemove = async (file:UploadFile) => {
        await axios.request({
            method: 'GET', 
            url: 'student/paper/answer/delete',
            params: {
                "answerId": file.uid
            }
        })
    }
    
    return (
        <div className="student_paper_detail">
            <Image.PreviewGroup
                preview={{
                    onChange: (current, prev) => console.log(`current index: ${current}, prev index: ${prev}`),
                }}
            >
                {
                    paperImages.map(item => (
                        <Image width={200} src={item.url} rootClassName='paper_image' key={item.uid}/>
                    ))
                }
            </Image.PreviewGroup>
            <div>
                <h3>请上传自己的答案</h3>
                <ImageUpload 
                    data={{paperId, "username": window.sessionStorage.getItem("username")}} 
                    url={window.location.origin + '/api/student/answer/imageUpload'} 
                    showUploadButton={score<0}
                    fileList={answers}
                    onFileChange={handleAnswerChange}
                    handleFileRemove={handlePhotoRemove}
                /> 
            </div>
            {
                score < 0 ? (
                <div className="button_group">
                    <Popconfirm
                        title="提醒"
                        description="确定要提交答案, 不检查一下？"
                        open={open}
                        onConfirm={()=>waitingForScore()}
                        onCancel={()=>setOpen(false)}
                        >
                        <Button type="primary" onClick={()=>setOpen(true)}>
                            提交答案
                        </Button>
                    </Popconfirm>
                </div>
                ) : (
                    <div>
                        <h3>分数</h3>
                        <p>你的分数是 {score}</p>
                    </div>
                )
            }
            {contextHolder}
        </div>
        
    )
}

export default PaperDetail