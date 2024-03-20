import { Button, Tree, InputNumber, Modal, Input, message, Alert,Space, Select } from 'antd';
import React, { useEffect, useState } from 'react';
import range from '@/util/range';
import numToCN from '@/util/numToCN';
import './AddPaperBoard.less'
import axios from 'axios';
import ImageUpload from '@/components/ImageUpload/ImageUpload';
import type { UploadFile } from 'antd/es/upload/interface';
const {TextArea} = Input;
/**
 * 配置试卷
 * - 大题数目
 * - 每道大题有多少道小题
 */
enum ModalContentState {
    closed = 0,
    setBigProblemNumber,
    setBigProblemType,
    setSmallProblemNumber,
    setProblemAnswer
}
const AddPaperBoard: React.FC = () => {
    // 上传答案相关
    const [isAnsModalOpen, setisAnsModalOpen] = useState(ModalContentState.closed);
    const [bigProblemNumber, setBigProblemNumber] = useState(1);
    const [bigProblemTypeList, setBigProblemTypeList] = useState([]);
    const [smallProblemNumberPerBigProblemList, setSmallProblemNumberPerBigProblemList] = useState([]);
    const [answers, setAnswers] = useState([]);
    const [treeData, setTreeData] = useState([]);

    // 重要参数
    const [paperId, setPaperId] = useState(-1)
    const [isSavePaper,setIsSavePaper] = useState(false);
    const [isSavePaperName, setIsSavePaperName] = useState(false);
    const [paperName, setPaperName] = useState("")
    const [paperPhotos, setPaperPhotos] = useState<UploadFile[]>([]);

    const init = async() =>  {
        if(paperId === -1) {
            const result = await axios.request({
                url: 'paper/add',
                method: 'GET',
                params: {
                    username: window.sessionStorage.getItem('username')
                }
            })
            console.log(result)
            if(result.data.msg === 'success') {
                // console.log("aaa ", result.data.data.paperId)
                setPaperId(result.data.data.paperId)
                // console.log("paperId", paperId)
            }
        }
    }

    const removePaper = async () => {
        await axios.request({
            url: 'paper/delete',
            method:'GET',
            params: {
                paperId
            }
        })
    }

    useEffect(()=>{
        init() 
        // return () => {
        //     if (!isSavePaper && paperId !== -1) {
        //         removePaper() 
        //     }
        // }
    })

    /**
     *  Modal 配置函数
     *  */ 
    
    // 打开Modal
    const showModal = () => {
        setisAnsModalOpen(ModalContentState.setBigProblemNumber)
    };

    // 点击ok, 当前值 + 1
    const handleAnsModalOk = () => {
        if(isAnsModalOpen === ModalContentState.setBigProblemNumber && bigProblemNumber !== smallProblemNumberPerBigProblemList.length) {
            setBigProblemTypeList(Array(bigProblemNumber).fill("xzt"))
            // 初始化小题的数组
            setSmallProblemNumberPerBigProblemList(Array(bigProblemNumber).fill(1))
        }
        else if (isAnsModalOpen === ModalContentState.setSmallProblemNumber) {
            //判断是否要重新设置数组
            let flag = false;
            if(answers.length === bigProblemNumber){
                for(let k = 0; k<bigProblemNumber;k++) {
                    if(smallProblemNumberPerBigProblemList[k] !== answers[k].length) {
                        flag = true;
                        break;
                    }
                }
            }
            else {
                flag = true;
            }

            // 初始化答案的二维数组
            if(flag) {
                let a = [];
                console.log(a)
                for(let i = 0; i<bigProblemNumber;i++) {
                    a.push([])
                    for(let j = 0; j < smallProblemNumberPerBigProblemList[i]; j++) {
                        a[i].push("");
                    }
                }
                console.log(a)
                setAnswers(a);
            }
        }
        else if(isAnsModalOpen === ModalContentState.setProblemAnswer){
            // 请求接口，将数据存入数据库
            axios.request({
                url:"paper/answer/add",
                method:'POST',
                data: {"answerList": answers, "paperId": paperId, "typeList": bigProblemTypeList}
            })
        }
        setisAnsModalOpen(current=> current + 1 > ModalContentState.setProblemAnswer ? ModalContentState.closed : current + 1);
    };

    // 取消，设置modal为关闭状态
    const handleAnsModalCancel = () => {
        setisAnsModalOpen(ModalContentState.closed);
    };
    
    // 根据isAnsModalOpen的不同返回不同内容
    const ModalContent = () => {
        switch (isAnsModalOpen) {
            case ModalContentState.setBigProblemNumber: return setBigProblemNumberModalContent()
            case ModalContentState.setBigProblemType: return setBigProblemTypeModalContent()
            case ModalContentState.setSmallProblemNumber: return setSmallProblemNumberModalContent()
            case ModalContentState.setProblemAnswer: return setProblemAnswerModalContent()
            default: return ''
        }
    }

    // 设置大题数目的内容
    const setBigProblemNumberModalContent = () => {
       return (
            <div>
                <span style={{marginRight: '2px'}}>一共有</span>
                    <InputNumber size="small" value={bigProblemNumber} min={1} onChange={onBigProblemNumberChange}/>
                <span style={{marginLeft: '2px'}}>道大题</span>
            </div>
       ) 
    }
    // 监听大题输入框的数字变化
    const onBigProblemNumberChange = (value: number) => {
        setBigProblemNumber(value)
        
    }

    // 设置大题题型的内容
    const setBigProblemTypeModalContent = () => {
        return (
            <div>
                <p>设置每道大题的题型</p>
                <ol style={{display: 'grid', gridTemplateColumns: '30% 30% 30%'}}>
                    {range({end: bigProblemNumber}).map((n,index)=><li key={index} style={{marginBottom: '12px',marginRight:'5px'}}>
                        <Select defaultValue={bigProblemTypeList[n]} style={{width: 100}} onChange={value=>handleBigProblemTypeChange(value,n)} options={[
                            {value: 'xzt', label: '选择题'},
                            {value: 'tkt', label: '填空题'},
                            {value: 'zwt', label: '作文题'}
                        ]}/>
                    </li>
                    )}
                </ol>
            </div>
        )
    }

    const handleBigProblemTypeChange = (value: string, index: number) => {
        let tempArray = bigProblemTypeList;
        tempArray[index] = value;
        console.log(tempArray);
        setBigProblemTypeList(tempArray);
    }

    // 设置小题数量的内容
    const setSmallProblemNumberModalContent = () => {
        return (
            <div>
                <p>设置每道大题的小题数量。 如果没有小题, 直接填1即可</p>
                <ol style={{display: 'grid', gridTemplateColumns: '30% 30% 30%'}}>
                    {range({end: bigProblemNumber}).map((n,index)=><li key={index} style={{marginBottom: '12px'}}><InputNumber size="small" min={1} defaultValue={smallProblemNumberPerBigProblemList[n]} onChange={value => onSmallProblemNumberChange(value,n)}/></li>)}
                </ol>
            </div>
        )
    }
    // 监听小题输入框的输入
    const onSmallProblemNumberChange = (value: number, index:number) => {
        let tempArray = smallProblemNumberPerBigProblemList;
        tempArray[index] = value;
        setSmallProblemNumberPerBigProblemList(tempArray);
    }

    const setProblemAnswerModalContent = () => {
        return (
            <div>
                <p>设置每道题的答案</p>
                <ol>
                    {range({end: bigProblemNumber}).map(i=> (
                        <li>
                            {smallProblemNumberPerBigProblemList[i] > 1 ? (
                                <div>
                                    <p>请输入第{numToCN(i+1)}题答案</p>
                                    {range({end: smallProblemNumberPerBigProblemList[i]}).map((j,index)=>(
                                        <div key={index}>
                                            <span>{i+1}.{j+1}</span>
                                            {bigProblemTypeList[i]==='zwt'
                                                ?<TextArea rows={4} placeholder="请输入对应的答案" defaultValue={answers[i][j]} onChange={e=>onAnswerChange(e.target.value,i,j)}/>
                                                :<Input placeholder="请输入对应的答案" defaultValue={answers[i][j]} onChange={e=>onAnswerChange(e.target.value,i,j)}/>
                                            }
                                        </div>
                                    ))}
                                </div>
                                ) : (
                                <div>
                                    <p>请输入第{numToCN(i+1)}题答案</p>
                                    {bigProblemTypeList[i]==='zwt'
                                        ?<TextArea rows={4} placeholder="请输入对应的答案" defaultValue={answers[i][0]} onChange={e=>onAnswerChange(e.target.value,i,0)}/>
                                        :<Input placeholder="请输入对应的答案" defaultValue={answers[i][0]} onChange={e=>onAnswerChange(e.target.value,i,0)}/>
                                    }
                                </div>
                            )}
                        </li>
                    ))}
                </ol>
            </div>
        )
    }

    const onAnswerChange = (value: string, i:number, j:number) => {
        console.log(value);
        let a = answers;
        a[i][j] = value;
        setAnswers(a);
        // 设置treeData
        setTreeData(generateTreeData());
    }

    const generateTreeData = () => {
        let treeData = [];
        for(let i = 0; i<bigProblemNumber;i++) {
            treeData.push({title: '第' + numToCN(i+1) + '题答案'})
            let a = [];
            for (let j = 0; j < smallProblemNumberPerBigProblemList[i];j++) {
                a.push({title: answers[i][j]})
            }
            treeData[treeData.length-1] = {...treeData[treeData.length-1], 'children': a}
        }
        return treeData;
    }
    
    // 保存试卷
    const savePaper = () => {
        if(paperName !== "") {
            if(!isSavePaperName) savePaperName()
            setIsSavePaper(true)
            message.success("上传成功")
        }
        else {
            message.error("请保存试卷名字")
        }
    }

    const savePaperName = async() => {
        console.log("papername", paperName)
        if(paperName === ""){
            message.error("试卷名字不能为空")
            return
        }
        const result = await axios.request({
            url: 'paper/name/update',
            method: 'GET',
            params: {
                paperId,
                paperName
            }
        })
        if(result.data.msg === 'success') {
            message.success("试卷名字保存成功")
            setIsSavePaperName(true)
        }
    }

    const handlePaperPhotosChange = (fileList:UploadFile[]) => {
        setPaperPhotos(fileList)
    }

    return (
        <div className='teacher_add_paper_board_body'>
            <h2>请上传你的试卷</h2>
            <Alert message="完成试卷相关信息的填写后, 记得点击上传按钮 " type="warning" />
            <div style={{height: '15px'}}></div>
            <Space.Compact style={{width: '100%'}}>
                <Input addonBefore="试卷名称" value={paperName} onChange={e=>{setPaperName(e.target.value)}}/> 
                <Button type='primary' onClick={()=>{savePaperName();}}>保存试卷名字</Button>
            </Space.Compact>
            <div className='photo_container'>
                <h4>上传试卷图片</h4>
                <ImageUpload 
                    data={{paperId}} 
                    url={window.location.origin + '/api/upload/imageUpload'} 
                    showUploadButton={true}
                    fileList={paperPhotos}
                    onFileChange={handlePaperPhotosChange}
                />
            </div>
            <div className='content_container'>
                <h4>上传题目答案</h4>
                <Button type="primary" onClick={showModal}>上传题目答案</Button>
                <div className='answer_container'>
                    <p className='tip'>试题答案</p>
                    <Tree
                        showLine={true}
                        treeData={treeData}
                    />
                </div>
                <Modal title="配置试卷答案" open={isAnsModalOpen > 0} onOk={handleAnsModalOk} onCancel={handleAnsModalCancel} width={isAnsModalOpen === ModalContentState.setProblemAnswer ? 1000 : 520}>
                   {ModalContent()} 
                </Modal>
            </div>
            <div style={{display: 'flex', justifyContent: 'center'}}>
                <Button type="primary" onClick={()=>savePaper()}>确定上传试卷</Button>
            </div>
        </div>
    )
}
export default AddPaperBoard