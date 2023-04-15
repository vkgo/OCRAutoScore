import React,{useState, useEffect} from 'react';
import { Image, Upload } from 'antd'
import { PlusOutlined} from '@ant-design/icons';
import type { UploadFile } from 'antd/es/upload/interface';
import { useParams } from "react-router-dom";
import './PaperDetail.less'
import axios from 'axios';
const PaperDetail: React.FC = () => {
    const paperId =  parseInt(useParams<{id: string}>()["id"])
    const [fileList, setFileList] = useState<UploadFile[]>([]);
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
                <Upload listType="picture-card" fileList={fileList}>
                    <div>
                        <PlusOutlined />
                        <div style={{ marginTop: 8 }}>Upload</div>
                    </div>
                </Upload>
            </div>
        </div>
        
    )
}

export default PaperDetail