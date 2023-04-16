import React,{useState, useEffect} from 'react';
import { Image } from 'antd'
import { useParams } from "react-router-dom";
import './PaperDetail.less'
import axios from 'axios';
import ImageUpload from '@/components/ImageUpload/ImageUpload';
const PaperDetail: React.FC = () => {
    const paperId =  parseInt(useParams<{id: string}>()["id"])
    const [paperImages, setImageList] = useState([])
    

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

    useEffect(()=>{
        if(paperImages.length ===  0) getPaperPhotos()
    })
    
    return (
        <div className="student_paper_detail">
            <Image.PreviewGroup
                preview={{
                    onChange: (current, prev) => console.log(`current index: ${current}, prev index: ${prev}`),
                }}
            >
                {
                    paperImages.map(item => (
                        <Image width={200} src={item.imgUrl} rootClassName='paper_image' key={item.id}/>
                    ))
                }
            </Image.PreviewGroup>
            <div>
                <h3>请上传自己的答案</h3>
                <ImageUpload data={{paperId, "username": window.sessionStorage.getItem("username")}} url={window.location.origin + '/api/student/answer/imageUpload'}/> 
            </div>
        </div>
        
    )
}

export default PaperDetail