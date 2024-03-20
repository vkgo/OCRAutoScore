import PaperList from '@/components/PaperList/PaperList';
import React, {useState, useEffect} from 'react';
import axios from 'axios';
const PaperLibrary: React.FC = () => {
    const [papers, setPapers] = useState([])
    useEffect(()=>{
        getPaperList()
    }, [])
    const getPaperList = async () => {
        const result = await axios.request({
            url: 'student/papers',
            method: 'GET'
        })
        if(result.data.msg === 'success') {
            setPapers(result.data.papers)
        }
    }
    return (
        <div>
            <h2>题库</h2>
            <PaperList baseUrl='/student/papers/detail/' list={papers} buttonText='点击作答' showDeleteButton={false}/>
        </div>
    )
}

export default PaperLibrary;