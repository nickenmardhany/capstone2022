import React from "react";
import { useContext } from "react";
import { useState } from "react";

import { UserContext } from "../context/UserContext";
import ErrorMessage from "./ErrorMessage";
import SuccessMessage from "./SuccessMessage";

const Register = () => {
    const [email, setEmail] = useState("");
    const [first_name, setFirstName] = useState("");
    const [last_name, setLastName] = useState("");
    const [username, setUsername] = useState("");
    const [password, setPassword] = useState("");
    const [errorMessage, setErrorMessage] = useState("");
    const [successMessage, setSuccessMessage] = useState("");
    const [,setToken] = useContext(UserContext);


    const submitRegistration = async () => {
        const requestOptions = {
            method: "POST",
            headers: { "Content-Type": "application/json"},
            body: JSON.stringify({first_name: first_name,last_name: last_name,username: username,email: email, password:password})

        };

        const response = await fetch("/register", requestOptions);
        const data = await response.json();

        if(!response.ok){
            setErrorMessage(data.detail);
        } 
        else {
            setSuccessMessage('Berhasil mendaftarkan akun');
            setToken(data.token)
        }
        


    };
    const handleSubmit = (e) => {
        e.preventDefault();
        if (password.length > 5) {
            submitRegistration();
        } else {
            setErrorMessage(
                "Ensure password greater than 5 characters"
            );
        }
    };

    return(
        <div className="column">
            <form action="" className="box" onSubmit={handleSubmit}>
                <h1 className="title has-text-centered">Register</h1>
                <div className="field">
                    <label className="label">First Name</label>
                    <div className="control">
                        <input type="text" placeholder="Enter your first name" value={first_name} onChange={(e)=> setFirstName(e.target.value)} className="input" required />
                    </div>
                </div>
                <div className="field">
                    <label className="label">Last Name</label>
                    <div className="control">
                        <input type="text" placeholder="Enter your last name" value={last_name} onChange={(e)=> setLastName(e.target.value)} className="input" required />
                    </div>
                </div>
                <div className="field">
                    <label className="label">Username</label>
                    <div className="control">
                        <input type="text" placeholder="Enter username" value={username} onChange={(e)=> setUsername(e.target.value)} className="input" required />
                    </div>
                </div>
                <div className="field">
                    <label className="label">Email Address</label>
                    <div className="control">
                        <input type="email" placeholder="Enter email" value={email} onChange={(e)=> setEmail(e.target.value)} className="input" required />
                    </div>
                </div>
                <div className="field">
                    <label className="label">Password</label>
                    <div className="control">
                        <input type="password" placeholder="Enter password" value={password} onChange={(e)=> setPassword(e.target.value)} className="input" required />
                    </div>
                </div>

                <ErrorMessage message={errorMessage}/>
                <SuccessMessage message={successMessage}/>
                <br/>
                <button className="button is-primary" type="submit">Register</button>
                
            </form>
        
        </div>
    )
};

export default Register;