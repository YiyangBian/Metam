import { useContext, useEffect, useState } from "react"
import { useNavigate, useLocation } from 'react-router-dom'
import TableCard from '../../components/TableCard'
import MainTable from '../../components/MainTable'
import Papa from 'papaparse'
import FormDataCtxt from "../../utils/formData"
import * as go from 'gojs';
import { ReactDiagram } from 'gojs-react';
import * as React from 'react';

import {Apptest} from "../../pages/Appjs/Apptest";

const Input = () => {

    let location = useLocation();
    let navigate = useNavigate();

    const [formData, setFormData] = useContext(FormDataCtxt);

    const handleFormChange = (field, value) => {
        setFormData(current => {
            return {
                ...current,
                [field]: value
            }
        })
    } 
    const { task, 
        attribute, 
        classification, 
        utilityMetric, 
        dataset, 
        matches, folder,file } = formData

    useEffect(() => {
        console.log(formData)
        console.log("file details")
        if (formData.file) {
            Papa.parse(formData.file, {
                header: true,
                skipEmptyLines: true,
                complete: r => fetchMatchData(r)//handleFormChange('dataset', r)
            })
            //fetchMatchData(formData.file)
            console.log(formData)

        }
    }, [formData.file])
    const fetchMatchData = (fileData) => {
        handleFormChange('dataset', fileData)
        console.log(formData.file.name)
        console.log(fileData)
        //console.log(JSON.stringify({'file':fData}))
        const url = '/api/tables'
        fetch(url, {
                method: 'post',
                body : JSON.stringify({'file':formData.file.name,'filedata':fileData})
            })
        .then(response => response.json())
        .then(data => handleFormChange('matches', data))
    }
    const handleFile = e => {  
        handleFormChange('file', e.target.files[0])
        
        console.log(formData.dataset)
    }
    
    const handleTaskChange = e => {
        handleFormChange('task', e.target.value);
        if (e.target.value !== 1) {
            handleFormChange('classification', null);
        }
        else {
            handleFormChange('classification', 1);
        }
    }

    const handleClassificationChange = e => {
        handleFormChange('classification', e.target.value);
    }

    const handleAttributeChange = e => {
        handleFormChange('attribute', e.target.value);
    }

    const handleUtilityMetricChange = e => {
        handleFormChange('utilityMetric', e.target.value);
    }

    const handleSubmit = () => {
        navigate("/results");
    }
    const showTables = () => {
        navigate("/tables");
    }
    const taskSubmit = () => {
        navigate("/taskoutput");
    }
    const Test = (color1,color2,color3,color4)=>{
        return Apptest(color1,color2,color3,color4);
    }
    return (
        <div className="container">
            <h1>Metam</h1>
            <div style={{width: "100%"}}>
            {
            formData.file? 
             <div style={{float:"left", width: "50%", backgroundColor:"#FFCCCB", margin: "0px 25px 0px 0px"}}>
                {
                    <div>Uploaded: {formData.file.name}</div> 
                }
                <span>
                    {
                        formData.file && <Form2 
                            formData={formData}
                            handleFormChange={handleFormChange}
                            handleSubmit={handleSubmit}
                        />
                    }
                </span>
            </div>
             : 
                <div style={{float:"left", width: "50%"}}>
                {
                    
                     <div style={{ width: "200%", textAlign:"center"}}>
                    <br></br>
                    <br></br>
                    <br></br>
                    <br></br>
                    <br></br>
                     <label><h4>Upload your csv dataset:</h4></label>
                    <input type="file" onChange={handleFile}></input>
                    <br></br>
                    <br></br>
                    <br></br>
                    <br></br>
                    <br></br>
                    <br></br>
                    <br></br>
                    <br></br>
                    <br></br>
                    </div>
                 }
                </div>
            }
            <div style={{ float:"right"}}>
                {
                formData.file ? <div>
                 <label>Choose the task to be performed on the data:</label>
                <select class="u-full-width" id="utility" style={{width:"45%"}} key="{task}task" value={task} onChange={handleTaskChange}>
                    <option value="Classification">Classification</option>
                    <option value="Regression">Regression</option>
                    <option value="What-if">What-if analysis</option>
                    <option value="How-to">How-to analysis</option>
                    <option value="5" disabled={true}>Your own function</option>
                </select>

                {
                //   (task === 1) && (
                //        <>
                 //           <label>Classification kind:</label>
                 //           <select class="u-full-width"  style={{width:"45%"}} id="utility" value="{classification}class" onChange={handleClassificationChange}>
                 //               <option value="1">Binary Classification</option>
                 //               <option value="2">Multi-label Classification</option>
                 //               <option value="3">Multi-class Classification</option>
                 //               <option value="4">Imbalanced Classification</option>
                 //           </select>
                 //       </> 
                //  )
                }

                <label>Choose the attribute to be measured:</label>
                <select class="u-full-width" id="attribute" style={{width:"45%"}} key="{attribute}attr" value={attribute} onChange={handleAttributeChange}>
                    {
                        dataset && (
                            dataset.meta.fields.map(f => <option value={f}>{f}</option>)
                        )   
                    }
                </select>

                <label>Choose the task utility metric:</label>
                <select class="u-full-width" id="utility" style={{width:"45%"}} key="{utilityMetric}util" value={utilityMetric} onChange={handleUtilityMetricChange}>
                    <option value="1">Mean Squared Error</option>
                    <option value="2">Mean Absolute Error</option>
                    <option value="4">F-score</option>
                    <option value="5" disabled={true}>Your own function</option>
                </select>
                <br></br>
                <br></br>
                

                </div> : <p></p>
                }
                </div>
                </div>
                <div style={{clear:"both"}}></div>
                <button type="submit" onClick={taskSubmit}>Run Task</button>
                <div>{formData.file ?
                Test('lightgreen','lightyellow','grey','grey'): Test('lightyellow','grey','grey','grey')}
                </div>
            
                

        </div>
    )
}

const Form2 = ({ formData, 
    handleFormChange, 
    handleSubmit }) => {

    const { task, 
        attribute, 
        classification, 
        utilityMetric, 
        dataset, 
        matches, folder,file } = formData

    
    const handleFolder = e => {  
        console.log(e.target.files,e.target.files[0].webkitRelativePath)
        handleFormChange('folder', e.target.files[0].webkitRelativePath)

        //iterate over the list and identify filenames
        //calculate folder name too
        handleFormChange('filelst', e.target.files)
        console.log(e.target.files[0].webkitRelativePath,formData.folder)
    }
    const handleTaskChange = e => {
        handleFormChange('task', e.target.value);
        if (e.target.value !== 1) {
            handleFormChange('classification', null);
        }
        else {
            handleFormChange('classification', 1);
        }
    }

    const handleClassificationChange = e => {
        handleFormChange('classification', e.target.value);
    }

    const handleAttributeChange = e => {
        handleFormChange('attribute', e.target.value);
    }

    const handleUtilityMetricChange = e => {
        handleFormChange('utilityMetric', e.target.value);
    }

    return <>
        <div style={{backgroundColor:"#FFCCCB"}}>
        <div className="container" style={{backgroundColor:"white"}}>
            {
                  formData.dataset ? 
                  <p><MainTable 
                    name={file.name}
                    preview={dataset.data}
                /></p>:<p></p>
            }
        </div>
        </div>
        <div style={{backgroundColor:"white"}}>
        
        <br></br>
        <br></br>
        <br></br>
        <br></br>
    </div>
    </>
}

export default Input

//<label>Choose the Table Candidates to be considered by Metam:</label>
//       <div className="container">
//           {
//                matches.map(m => <TableCard 
//                    name={m.name}
//                    id={m.id}
//                    preview={m.preview} 
//                />)
 //           }
//        </div>
    