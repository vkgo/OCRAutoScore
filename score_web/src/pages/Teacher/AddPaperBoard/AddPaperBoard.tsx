import { Button, Upload, Tree, InputNumber, Modal, Input} from 'antd';
import { PlusOutlined} from '@ant-design/icons';
import type { UploadFile } from 'antd/es/upload/interface';
import React, { useState } from 'react';
import range from '@/util/range';
import numToCN from '@/util/numToCN';
import './AddPaperBoard.less'
const {TextArea} = Input;

/**
 * 配置试卷
 * - 大题数目
 * - 每道大题有多少道小题
 */
enum ModalContentState {
    closed = 0,
    setBigProblemNumber,
    setSmallProblemNumber,
    setProblemAnswer
}
const AddPaperBoard: React.FC = () => {
    const [fileList, setFileList] = useState<UploadFile[]>([]);
    const [isModalOpen, setIsModalOpen] = useState(ModalContentState.closed);
    const [bigProblemNumber, setBigProblemNumber] = useState(1);
    const [smallProblemNumberPerBigProblemList, setSmallProblemNumberPerBigProblemList] = useState([]);
    const [answers, setAnswers] = useState([]);
    const [treeData, setTreeData] = useState([]);
    /**
     *  Modal 配置函数
     *  */ 
    
    // 打开Modal
    const showModal = () => {
        setIsModalOpen(ModalContentState.setBigProblemNumber)
    };

    // 点击ok, 当前值 + 1
    const handleOk = () => {
        if(isModalOpen === ModalContentState.setBigProblemNumber && bigProblemNumber !== smallProblemNumberPerBigProblemList.length) {
            // 初始化小题的数组
            setSmallProblemNumberPerBigProblemList(Array(bigProblemNumber).fill(1))
        }
        else if (isModalOpen === ModalContentState.setSmallProblemNumber) {
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
        setIsModalOpen(current=> current + 1 > ModalContentState.setProblemAnswer ? ModalContentState.closed : current + 1);
    };

    // 取消，设置modal为关闭状态
    const handleCancel = () => {
        setIsModalOpen(ModalContentState.closed);
    };
    
    // 根据isModalOpen的不同返回不同内容
    const ModalContent = () => {
        switch (isModalOpen) {
            case ModalContentState.setBigProblemNumber: return setBigProblemNumberModalContent()
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
                                            <TextArea rows={4} placeholder="请输入对应的答案" defaultValue={answers[i][j]} onChange={e=>onAnswerChange(e.target.value,i,j)}/>   
                                        </div>
                                    ))}
                                </div>
                                ) : (
                                <div>
                                    <p>请输入第{numToCN(i+1)}题答案</p>
                                    <TextArea rows={4} defaultValue={answers[i][0]} onChange={e=>onAnswerChange(e.target.value,i,0)}/>
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

    return (
        <div className='teacher_add_paper_board_body'>
            <h2>请上传你的试卷</h2>
            <div className='photo_container'>
                <h4>上传试卷图片</h4>
                <Upload listType="picture-card" fileList={fileList}>
                    <div>
                        <PlusOutlined />
                        <div style={{ marginTop: 8 }}>Upload</div>
                    </div>
                </Upload>
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
                <Modal title="配置试卷答案" open={isModalOpen > 0} onOk={handleOk} onCancel={handleCancel} width={isModalOpen === ModalContentState.setProblemAnswer ? 1000 : 520}>
                   {ModalContent()} 
                </Modal>
            </div>
        </div>
    )
}
export default AddPaperBoard