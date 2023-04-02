import React,{useState} from 'react';
import { Image, Upload } from 'antd'
import { PlusOutlined} from '@ant-design/icons';
import type { UploadFile } from 'antd/es/upload/interface';
import './PaperDetail.less'
const PaperDetail: React.FC = () => {
    const [fileList, setFileList] = useState<UploadFile[]>([]);
    return (
        <div className="student_paper_detail">
            <Image.PreviewGroup
                preview={{
                    onChange: (current, prev) => console.log(`current index: ${current}, prev index: ${prev}`),
                }}
            >
                <Image width={200} src={require("@/assets/papers/paper1.jpg")} rootClassName='paper_image'/>
                <Image
                    width={200}
                    src={require("@/assets/papers/paper2.png")}
                    rootClassName='paper_image'
                />
                <Image
                    width={200}
                    src={require("@/assets/papers/paper3.png")}
                    rootClassName='paper_image'
                />

                <Image width={200} src={require("@/assets/papers/paper1.jpg")} rootClassName='paper_image'/>

                <Image width={200} src={require("@/assets/papers/paper1.jpg")} rootClassName='paper_image'/>

                <Image width={200} src={require("@/assets/papers/paper1.jpg")} rootClassName='paper_image'/>

                <Image width={200} src={require("@/assets/papers/paper1.jpg")} rootClassName='paper_image'/>
                
                <Image width={200} src={require("@/assets/papers/paper1.jpg")} rootClassName='paper_image'/>
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