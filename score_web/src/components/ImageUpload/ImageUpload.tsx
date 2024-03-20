import React,{useState} from 'react';
import { Upload, Modal, message} from 'antd'
import { PlusOutlined} from '@ant-design/icons';
import type { UploadFile } from 'antd/es/upload/interface';

interface ImageUploadPropsInterface {
    url: string,
    data: {paperId: number, username?: string},
    showUploadButton?: boolean,
    fileList: UploadFile[],
    onFileChange: (files: UploadFile[]) => void
    handleFileRemove ?: (files: UploadFile) => void
}

const ImageUpload:React.FC<ImageUploadPropsInterface> = (props) =>{
    const [imagePreviewOpen, setImagePreviewOpen] = useState(false);
    const [previewImage, setPreviewImage] = useState('');

    const handlePhotoPreview = (file:UploadFile) => {
        setPreviewImage(file.url||file.thumbUrl)
        setImagePreviewOpen(true)
    }

    const handlePhotoModalCancel = () => {
        setImagePreviewOpen(false)
    }

    const uploadButton = (
        <div>
            <PlusOutlined />
            <div style={{ marginTop: 8 }}>Upload</div>
        </div>
    );

    const handlePhotoChange = async ({ file, fileList }) => {
        if (file.status === 'done') { 
            // {status: 0, data: {name: 'xxx.jpg', url: '图片地址'}}
            const result = file.response
            console.log(result);
            console.log(fileList);
            if (result.msg === 'success') {
                message.success('上传图片成功')
                const { name, url } = result.data
                file = fileList[fileList.length - 1]
                file.name = name
                file.url = url
            } 
            
            else {
                message.error('上传图片失败')
            }
        } 
        props.onFileChange(fileList);
    } 

    return (
        <>
        <Upload 
            name={'upload_image'}
            accept="image/*" 
            action={props.url} 
            data = {props.data}
            listType="picture-card" 
            fileList={props.fileList}
            onPreview={handlePhotoPreview}  
            onChange={handlePhotoChange}
            onRemove={props.handleFileRemove}
        >
            {props.showUploadButton ? uploadButton : false}
        </Upload>
        <Modal open={imagePreviewOpen} footer={null} onCancel={handlePhotoModalCancel}>
            <img alt="example" style={{ width: '100%' }} src={previewImage} />
        </Modal>
        </>
    )
}

export default ImageUpload;