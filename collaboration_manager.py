# collaboration_manager.py

import streamlit as st
from datetime import datetime
from typing import Dict, List, Any, Optional
import json
import uuid

class CollaborationManager:
    """Manages collaboration features for The Analyst"""
    
    def __init__(self):
        if "collaboration_sessions" not in st.session_state:
            st.session_state.collaboration_sessions = {}
        if "shared_analyses" not in st.session_state:
            st.session_state.shared_analyses = {}
        if "comments" not in st.session_state:
            st.session_state.comments = {}
        if "analysis_history" not in st.session_state:
            st.session_state.analysis_history = []
    
    def create_collaboration_session(self, session_name: str, creator: str) -> str:
        """Create a new collaboration session"""
        session_id = str(uuid.uuid4())[:8]
        
        session_data = {
            'id': session_id,
            'name': session_name,
            'creator': creator,
            'created_at': datetime.now(),
            'participants': [creator],
            'conversation_history': [],
            'shared_files': [],
            'permissions': {'read': True, 'write': True, 'admin': True}
        }
        
        st.session_state.collaboration_sessions[session_id] = session_data
        return session_id
    
    def join_session(self, session_id: str, participant_name: str) -> bool:
        """Join an existing collaboration session"""
        if session_id in st.session_state.collaboration_sessions:
            session = st.session_state.collaboration_sessions[session_id]
            if participant_name not in session['participants']:
                session['participants'].append(participant_name)
            return True
        return False
    
    def add_comment(self, analysis_id: str, user_name: str, comment_text: str, 
                   comment_type: str = "general") -> str:
        """Add a comment to an analysis"""
        comment_id = str(uuid.uuid4())[:8]
        
        comment_data = {
            'id': comment_id,
            'analysis_id': analysis_id,
            'user': user_name,
            'text': comment_text,
            'type': comment_type,  # general, insight, question, suggestion
            'timestamp': datetime.now(),
            'replies': []
        }
        
        if analysis_id not in st.session_state.comments:
            st.session_state.comments[analysis_id] = []
        
        st.session_state.comments[analysis_id].append(comment_data)
        return comment_id
    
    def add_reply(self, comment_id: str, user_name: str, reply_text: str) -> bool:
        """Add a reply to a comment"""
        for analysis_id, comments in st.session_state.comments.items():
            for comment in comments:
                if comment['id'] == comment_id:
                    reply_data = {
                        'user': user_name,
                        'text': reply_text,
                        'timestamp': datetime.now()
                    }
                    comment['replies'].append(reply_data)
                    return True
        return False
    
    def save_analysis_snapshot(self, analysis_name: str, conversation_history: List[Dict],
                             file_info: Dict, analysis_results: Dict, user_name: str) -> str:
        """Save a snapshot of the current analysis"""
        snapshot_id = str(uuid.uuid4())[:8]
        
        snapshot_data = {
            'id': snapshot_id,
            'name': analysis_name,
            'creator': user_name,
            'created_at': datetime.now(),
            'conversation_history': conversation_history.copy(),
            'file_info': {k: v for k, v in file_info.items() if k != 'dataframe'},  # Exclude large data
            'analysis_results': analysis_results.copy(),
            'tags': [],
            'is_public': False
        }
        
        st.session_state.analysis_history.append(snapshot_data)
        return snapshot_id
    
    def load_analysis_snapshot(self, snapshot_id: str) -> Optional[Dict]:
        """Load a previously saved analysis snapshot"""
        for snapshot in st.session_state.analysis_history:
            if snapshot['id'] == snapshot_id:
                return snapshot
        return None
    
    def share_analysis(self, snapshot_id: str, share_with: List[str], 
                      permissions: Dict[str, bool]) -> bool:
        """Share an analysis with other users"""
        for snapshot in st.session_state.analysis_history:
            if snapshot['id'] == snapshot_id:
                if snapshot_id not in st.session_state.shared_analyses:
                    st.session_state.shared_analyses[snapshot_id] = {
                        'shared_with': [],
                        'permissions': {}
                    }
                
                for user in share_with:
                    if user not in st.session_state.shared_analyses[snapshot_id]['shared_with']:
                        st.session_state.shared_analyses[snapshot_id]['shared_with'].append(user)
                    st.session_state.shared_analyses[snapshot_id]['permissions'][user] = permissions
                
                return True
        return False
    
    def get_user_analyses(self, user_name: str) -> List[Dict]:
        """Get all analyses accessible to a user"""
        user_analyses = []
        
        # Own analyses
        for snapshot in st.session_state.analysis_history:
            if snapshot['creator'] == user_name:
                user_analyses.append({
                    **snapshot,
                    'access_type': 'owner'
                })
        
        # Shared analyses
        for snapshot_id, share_info in st.session_state.shared_analyses.items():
            if user_name in share_info['shared_with']:
                snapshot = self.load_analysis_snapshot(snapshot_id)
                if snapshot:
                    user_analyses.append({
                        **snapshot,
                        'access_type': 'shared',
                        'permissions': share_info['permissions'].get(user_name, {})
                    })
        
        return sorted(user_analyses, key=lambda x: x['created_at'], reverse=True)
    
    def export_session_summary(self, session_id: str) -> Dict[str, Any]:
        """Export a summary of a collaboration session"""
        if session_id not in st.session_state.collaboration_sessions:
            return {}
        
        session = st.session_state.collaboration_sessions[session_id]
        
        # Get comments for this session
        session_comments = []
        for analysis_id, comments in st.session_state.comments.items():
            if analysis_id.startswith(session_id):
                session_comments.extend(comments)
        
        summary = {
            'session_info': session,
            'total_participants': len(session['participants']),
            'total_comments': len(session_comments),
            'duration': str(datetime.now() - session['created_at']),
            'key_insights': self._extract_key_insights(session_comments),
            'action_items': self._extract_action_items(session_comments)
        }
        
        return summary
    
    def _extract_key_insights(self, comments: List[Dict]) -> List[str]:
        """Extract key insights from comments"""
        insights = []
        
        for comment in comments:
            if comment['type'] == 'insight':
                insights.append(f"{comment['user']}: {comment['text']}")
        
        return insights[:10]  # Limit to top 10
    
    def _extract_action_items(self, comments: List[Dict]) -> List[str]:
        """Extract action items from comments"""
        actions = []
        
        for comment in comments:
            text = comment['text'].lower()
            if any(keyword in text for keyword in ['todo', 'action', 'next step', 'follow up']):
                actions.append(f"{comment['user']}: {comment['text']}")
        
        return actions[:10]  # Limit to top 10
    
    def render_collaboration_interface(self):
        """Render the collaboration interface"""
        st.subheader("👥 Collaboration Hub")
        
        # Get current user (simplified - in production this would be from authentication)
        if "current_user" not in st.session_state:
            st.session_state.current_user = st.text_input("Enter your name:", value="User")
        
        current_user = st.session_state.current_user
        
        if not current_user:
            st.warning("Please enter your name to use collaboration features.")
            return
        
        # Tabs for different collaboration features
        tab1, tab2, tab3, tab4 = st.tabs(["My Analyses", "Shared Work", "Comments", "Session Management"])
        
        with tab1:
            self._render_my_analyses(current_user)
        
        with tab2:
            self._render_shared_work(current_user)
        
        with tab3:
            self._render_comments_interface(current_user)
        
        with tab4:
            self._render_session_management(current_user)
    
    def _render_my_analyses(self, user_name: str):
        """Render user's personal analyses"""
        st.write("**📊 My Analyses**")
        
        user_analyses = self.get_user_analyses(user_name)
        own_analyses = [a for a in user_analyses if a['access_type'] == 'owner']
        
        if not own_analyses:
            st.info("No saved analyses yet. Complete an analysis and save a snapshot to see it here.")
            return
        
        for analysis in own_analyses[:10]:  # Show last 10
            with st.expander(f"📋 {analysis['name']} - {analysis['created_at'].strftime('%Y-%m-%d %H:%M')}"):
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.write(f"**Questions Asked:** {len([m for m in analysis['conversation_history'] if m['role'] == 'user'])}")
                
                with col2:
                    if st.button("Load", key=f"load_{analysis['id']}"):
                        # Load this analysis into current session
                        st.session_state.messages = analysis['conversation_history']
                        st.success("Analysis loaded!")
                        st.rerun()
                
                with col3:
                    if st.button("Share", key=f"share_{analysis['id']}"):
                        st.session_state.sharing_analysis = analysis['id']
        
        # Handle sharing
        if "sharing_analysis" in st.session_state:
            self._render_sharing_interface(st.session_state.sharing_analysis, user_name)
    
    def _render_shared_work(self, user_name: str):
        """Render analyses shared with the user"""
        st.write("**🤝 Shared with Me**")
        
        user_analyses = self.get_user_analyses(user_name)
        shared_analyses = [a for a in user_analyses if a['access_type'] == 'shared']
        
        if not shared_analyses:
            st.info("No analyses have been shared with you yet.")
            return
        
        for analysis in shared_analyses:
            with st.expander(f"🔗 {analysis['name']} (by {analysis['creator']})"):
                st.write(f"Shared on: {analysis['created_at'].strftime('%Y-%m-%d %H:%M')}")
                
                col1, col2 = st.columns(2)
                with col1:
                    if st.button("View", key=f"view_{analysis['id']}"):
                        st.session_state.messages = analysis['conversation_history']
                        st.success("Analysis loaded!")
                        st.rerun()
                
                with col2:
                    if st.button("Comment", key=f"comment_{analysis['id']}"):
                        st.session_state.commenting_on = analysis['id']
    
    def _render_comments_interface(self, user_name: str):
        """Render comments and discussions interface"""
        st.write("**💬 Comments & Discussions**")
        
        # Show recent comments
        all_comments = []
        for analysis_id, comments in st.session_state.comments.items():
            for comment in comments:
                comment['analysis_id'] = analysis_id
                all_comments.append(comment)
        
        # Sort by timestamp
        all_comments.sort(key=lambda x: x['timestamp'], reverse=True)
        
        if not all_comments:
            st.info("No comments yet. Add comments to your analyses to start discussions.")
            return
        
        # Show recent comments
        st.write("**Recent Comments:**")
        for comment in all_comments[:10]:
            analysis_name = "Unknown Analysis"
            # Try to find analysis name
            for snapshot in st.session_state.analysis_history:
                if snapshot['id'] == comment['analysis_id']:
                    analysis_name = snapshot['name']
                    break
            
            with st.container():
                st.write(f"**{comment['user']}** on *{analysis_name}*")
                st.write(f"🏷️ {comment['type'].title()}: {comment['text']}")
                st.caption(f"{comment['timestamp'].strftime('%Y-%m-%d %H:%M')}")
                
                # Show replies
                if comment['replies']:
                    for reply in comment['replies']:
                        st.write(f"  ↳ **{reply['user']}**: {reply['text']}")
                
                # Add reply option
                if st.button("Reply", key=f"reply_{comment['id']}"):
                    reply_text = st.text_input("Your reply:", key=f"reply_input_{comment['id']}")
                    if reply_text:
                        self.add_reply(comment['id'], user_name, reply_text)
                        st.success("Reply added!")
                        st.rerun()
                
                st.divider()
        
        # Add new comment interface
        if "commenting_on" in st.session_state:
            st.write("**Add Comment:**")
            comment_type = st.selectbox("Comment type:", ["general", "insight", "question", "suggestion"])
            comment_text = st.text_area("Your comment:")
            
            if st.button("Add Comment"):
                if comment_text:
                    self.add_comment(st.session_state.commenting_on, user_name, comment_text, comment_type)
                    st.success("Comment added!")
                    del st.session_state.commenting_on
                    st.rerun()
    
    def _render_session_management(self, user_name: str):
        """Render session management interface"""
        st.write("**🔧 Session Management**")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Create New Session**")
            session_name = st.text_input("Session name:")
            if st.button("Create Session") and session_name:
                session_id = self.create_collaboration_session(session_name, user_name)
                st.success(f"Session created! ID: {session_id}")
        
        with col2:
            st.write("**Join Existing Session**")
            session_id = st.text_input("Session ID:")
            if st.button("Join Session") and session_id:
                if self.join_session(session_id, user_name):
                    st.success("Joined session successfully!")
                else:
                    st.error("Session not found!")
        
        # Show current sessions
        st.write("**Active Sessions:**")
        user_sessions = [s for s in st.session_state.collaboration_sessions.values() 
                        if user_name in s['participants']]
        
        for session in user_sessions:
            with st.expander(f"📋 {session['name']} ({len(session['participants'])} participants)"):
                st.write(f"Created by: {session['creator']}")
                st.write(f"Participants: {', '.join(session['participants'])}")
                st.write(f"Created: {session['created_at'].strftime('%Y-%m-%d %H:%M')}")
                
                if st.button("Export Summary", key=f"export_{session['id']}"):
                    summary = self.export_session_summary(session['id'])
                    st.json(summary)
    
    def _render_sharing_interface(self, analysis_id: str, user_name: str):
        """Render interface for sharing an analysis"""
        st.write("**Share Analysis**")
        
        share_with = st.text_input("Share with (comma-separated usernames):")
        
        col1, col2 = st.columns(2)
        with col1:
            read_permission = st.checkbox("Read access", value=True)
        with col2:
            write_permission = st.checkbox("Write access", value=False)
        
        if st.button("Share"):
            if share_with:
                users = [u.strip() for u in share_with.split(',')]
                permissions = {'read': read_permission, 'write': write_permission}
                
                if self.share_analysis(analysis_id, users, permissions):
                    st.success(f"Analysis shared with {', '.join(users)}!")
                    del st.session_state.sharing_analysis
                    st.rerun()
                else:
                    st.error("Failed to share analysis.")
    
    def add_to_history(self, analysis_name: str, conversation_history: List[Dict],
                      file_info: Dict, analysis_results: Dict):
        """Automatically add completed analysis to history"""
        if "current_user" in st.session_state and st.session_state.current_user:
            self.save_analysis_snapshot(
                analysis_name, conversation_history, file_info, 
                analysis_results, st.session_state.current_user
            )